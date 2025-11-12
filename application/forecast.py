import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Stats to forecast
SKATER_STATS = ["G", "A", "PPP", "SOG", "FOW", "HIT", "BLK", "PIM"]


def _parse_toi_value(v) -> float:
    """Parse TOI strings from CSV into minutes as float.
    Handles examples like:
    - "0:20:12" -> 20 + 12/60
    - "0 days 00:20:53.285714286" -> 20 + 53/60
    - already numeric minutes
    Returns minutes (float). Invalid/missing -> np.nan
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v)
    if s == "" or s.lower() == "nan":
        return np.nan
    try:
        # pure numeric assumed minutes
        return float(s)
    except Exception:
        pass
    # Formats
    try:
        if "days" in s:
            # "0 days 00:20:53.285714286"
            parts = s.split()
            # last part like HH:MM:SS(.fractions)
            hms = parts[-1]
            hh, mm, ss = hms.split(":")
            minutes = int(hh) * 60 + int(mm) + float(ss) / 60.0
            return minutes
        if s.count(":") == 2:
            # H:MM:SS
            hh, mm, ss = s.split(":")
            minutes = int(hh) * 60 + int(mm) + float(ss) / 60.0
            return minutes
        if s.count(":") == 1:
            # MM:SS
            mm, ss = s.split(":")
            minutes = int(mm) + float(ss) / 60.0
            return minutes
    except Exception:
        return np.nan
    return np.nan


def _safe_weighted_mean(values: List[float], weights: List[float]) -> float:
    vals = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)
    mask = ~np.isnan(vals)
    if mask.sum() == 0:
        return np.nan
    w = w.copy()
    w[~mask] = 0.0
    if w.sum() == 0:
        # fall back to simple mean of available
        return np.nanmean(vals[mask])
    w = w / w.sum()
    return float(np.sum(vals * w))


def _expected_per_game(row: pd.Series, stat: str, season_weight: float, last7_weight: float, last_year_weight: float = 0.0) -> float:
    """Compute expected stat per game using per-60 rates and TOI per game.
    Blends Season-to-date, Last 7, and optionally Last Year if available.
    PPP uses PP TOI; others use all-situations TOI.
    Robust to missing inputs: falls back to all-situations TOI when PP TOI is missing;
    returns 0.0 if all components are unavailable.
    """
    # choose TOI columns based on stat
    if stat == "PPP":
        toi_cols = ("szn_TOI/GP_pp", "l7_TOI/GP_pp", "ly_TOI/GP_pp")
        # Fallbacks for PP TOI if missing: use all-situations TOI
        toi_fallbacks = ("szn_TOI/GP_all", "l7_TOI/GP_all", "ly_TOI/GP_all")
    else:
        toi_cols = ("szn_TOI/GP_all", "l7_TOI/GP_all", "ly_TOI/GP_all")
        toi_fallbacks = toi_cols

    rate_cols = (f"szn_{stat}", f"l7_{stat}", f"ly_{stat}")

    # Coerce rates to numeric
    szn_rate = pd.to_numeric(row.get(rate_cols[0], np.nan), errors="coerce")
    l7_rate = pd.to_numeric(row.get(rate_cols[1], np.nan), errors="coerce")
    ly_rate = pd.to_numeric(row.get(rate_cols[2], np.nan), errors="coerce")

    # Parse TOI minutes with fallbacks when primary is missing
    def _get_toi(primary_col: str, fallback_col: str) -> float:
        v = _parse_toi_value(row.get(primary_col, np.nan))
        if pd.isna(v):
            v = _parse_toi_value(row.get(fallback_col, np.nan))
        return v

    szn_toi_min = _get_toi(toi_cols[0], toi_fallbacks[0])
    l7_toi_min = _get_toi(toi_cols[1], toi_fallbacks[1])
    ly_toi_min = _get_toi(toi_cols[2], toi_fallbacks[2])

    # convert per60 rate to per-game expectation using TOI minutes
    szn_pg = np.nan if pd.isna(szn_rate) or pd.isna(szn_toi_min) else float(szn_rate) * (float(szn_toi_min) / 60.0)
    l7_pg = np.nan if pd.isna(l7_rate) or pd.isna(l7_toi_min) else float(l7_rate) * (float(l7_toi_min) / 60.0)
    ly_pg = np.nan if pd.isna(ly_rate) or pd.isna(ly_toi_min) else float(ly_rate) * (float(ly_toi_min) / 60.0)

    out = _safe_weighted_mean([szn_pg, l7_pg, ly_pg], [season_weight, last7_weight, last_year_weight])
    # Ensure we don't propagate NaNs to output CSV; fall back to 0.0 when unavailable
    return 0.0 if pd.isna(out) else out


def _sos_multiplier(sos_frac: float, sos_weight: float) -> float:
    """Map SOS fraction (0..1, where 1 is hardest) to a scaling multiplier.
    Center at 0.5: harder schedules reduce expectations, easier increase.
    multiplier = 1 + sos_weight*(0.5 - sos)
    """
    if sos_frac is None or np.isnan(sos_frac):
        return 1.0
    return 1.0 + float(sos_weight) * (0.5 - float(sos_frac))


def _parse_sos(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    if not s:
        return np.nan
    if s.endswith('%'):
        try:
            return float(s[:-1]) / 100.0
        except Exception:
            return np.nan
    try:
        f = float(s)
        # assume already 0..1 if <=1, else percent 0..100
        return f if f <= 1.0 else f / 100.0
    except Exception:
        return np.nan


def _lookup_games_and_sos(lookup: pd.DataFrame, team: str, current_week: int) -> Tuple[int, int, int, float, float, float]:
    """Return (games_row, games_next, games_ros, sos_row, sos_next, sos_ros_weighted)
    - games_row = GamesRestOfWeek for current_week row
    - games_next = Games for next week (current_week+1)
    - games_ros = GamesROS for current_week row
    - sos_row = SOS for current_week row (fraction)
    - sos_next = SOS for next week row (fraction)
    - sos_ros_weighted = weighted average SOS across remaining weeks (weights = Games)
    """
    team_rows = lookup[lookup['TM'] == team]
    if team_rows.empty:
        return 0, 0, 0, np.nan, np.nan, np.nan
    cur = team_rows[team_rows['Week'] == current_week]
    nxt = team_rows[team_rows['Week'] == current_week + 1]

    games_row = int(cur['GamesRestOfWeek'].iloc[0]) if not cur.empty else 0
    games_ros = int(cur['GamesROS'].iloc[0]) if not cur.empty else 0
    sos_row = _parse_sos(cur['SOS'].iloc[0]) if not cur.empty else np.nan

    games_next = int(nxt['Games'].iloc[0]) if not nxt.empty else 0
    sos_next = _parse_sos(nxt['SOS'].iloc[0]) if not nxt.empty else np.nan

    # ROS weeks = (current_week+1 .. max)
    future = team_rows[team_rows['Week'] > current_week]
    if not future.empty:
        sos_vals = future['SOS'].apply(_parse_sos).values.astype(float)
        games_vals = future['Games'].values.astype(float)
        if games_vals.sum() > 0:
            sos_ros = float(np.sum(sos_vals * games_vals) / np.sum(games_vals))
        else:
            sos_ros = np.nan
    else:
        sos_ros = np.nan

    return games_row, games_next, games_ros, sos_row, sos_next, sos_ros


def forecast(
    skaters_csv: str = os.path.join("data", "merged_skaters.csv"),
    lookup_csv: str = os.path.join("data", "lookup_table.csv"),
    out_csv: str = os.path.join("data", "forecasts.csv"),
    current_week: int = 1,
    season_weight: float = 0.5,
    last7_weight: float = 0.2,
    last_year_weight: float = 0.3,
    sos_weight: float = 0.3,
    horizons: Tuple[str, ...] = ("row", "next", "ros"),
    auto_weights: bool = True,
    weeks_in_season: int = 24,
) -> str:
    """Compute forecasted expected counts for requested horizons and save to CSV.

    If auto_weights is True (default), season_weight and last_year_weight are computed via a
    sliding scale over the season with last7_weight held constant:
      - Define week_index = clamp((current_week - 1) / (weeks_in_season - 1), 0..1)
      - The remaining weight mass = (1.0 - last7_weight)
      - last_year_weight = remaining * (1 - week_index)
      - season_weight   = remaining * week_index
    This gradually shifts weight from last_year to current-season actuals as weeks progress.

    Returns path to the output CSV.
    """
    sk = pd.read_csv(skaters_csv)
    lu = pd.read_csv(lookup_csv)

    # Auto-attach prior-season stats if available to provide ly_* columns by default
    try:
        prior_path = os.path.join(os.path.dirname(skaters_csv), "merged_skaters_prior.csv")
        if os.path.exists(prior_path):
            prior = pd.read_csv(prior_path)
            # Build normalized key for merge
            def _norm_name(x: str) -> str:
                s = str(x or "").strip().lower()
                for ch in [".", ",", "'", '"', "`"]:
                    s = s.replace(ch, "")
                return " ".join(s.split())
            sk["_key_player"] = sk.get("Player").apply(_norm_name)
            prior["_key_player"] = prior.get("Player").apply(_norm_name)
            # Map prior szn_* per-60 rates to ly_* expected names
            rate_map = {f"szn_{s}": f"ly_{s}" for s in SKATER_STATS}
            # PPP is included in SKATER_STATS already
            toi_map = {"szn_TOI/GP_all": "ly_TOI/GP_all", "szn_TOI/GP_pp": "ly_TOI/GP_pp"}
            keep_cols = ["_key_player"] + [c for c in list(rate_map.keys()) + list(toi_map.keys()) if c in prior.columns]
            prior_sub = prior[keep_cols].copy()
            prior_sub.rename(columns={**rate_map, **toi_map}, inplace=True)
            # Deduplicate prior by player key
            prior_sub = prior_sub.groupby("_key_player", as_index=False).first()
            # Merge into current
            sk = sk.merge(prior_sub, on="_key_player", how="left")
    except Exception:
        # Non-fatal: proceed without prior if any issue
        pass

    # Normalize columns used for filters and keys
    for col in ("Team", "team", "TM"):
        if col in sk.columns:
            break
    # Filter to owned teams (team_name not null)
    if 'team_name' in sk.columns:
        sk = sk[sk['team_name'].notna() & (sk['team_name'].astype(str).str.strip() != "")].copy()

    # Restrict lookup columns
    required_lu_cols = ['TM', 'Week', 'Games', 'SOS', 'GamesRestOfWeek', 'GamesROS']
    missing = [c for c in required_lu_cols if c not in lu.columns]
    if missing:
        raise ValueError(f"lookup_table.csv missing columns: {missing}")

    # Determine weights (optionally sliding scale)
    if auto_weights:
        try:
            denom = max(1, int(weeks_in_season) - 1)
        except Exception:
            denom = 23
        week_index = (int(current_week) - 1) / float(denom)
        week_index = max(0.0, min(1.0, week_index))
        remaining = max(0.0, 1.0 - float(last7_weight))
        season_w_eff = remaining * week_index
        last_year_w_eff = remaining * (1.0 - week_index)
    else:
        season_w_eff = float(season_weight)
        last_year_w_eff = float(last_year_weight)
    last7_w_eff = float(last7_weight)

    # Build forecasts per player
    rows = []
    for _, r in sk.iterrows():
        team = str(r.get('Team') or r.get('TM') or r.get('team') or '').strip()
        if not team:
            continue
        games_row, games_next, games_ros, sos_row, sos_next, sos_ros = _lookup_games_and_sos(lu, team, int(current_week))

        # per-game expectations for each stat
        per_game: Dict[str, float] = {}
        for stat in SKATER_STATS:
            per_game[stat] = _expected_per_game(r, stat, season_w_eff, last7_w_eff, last_year_w_eff)

        # SOS multipliers per horizon
        m_row = _sos_multiplier(sos_row, sos_weight)
        m_next = _sos_multiplier(sos_next, sos_weight)
        m_ros = _sos_multiplier(sos_ros, sos_weight)

        out_row = {
            'Player': r.get('Player'),
            'Team': team,
            'Position': r.get('Position') or r.get('Elig_Pos') or r.get('pos_group'),
            'team_name': r.get('team_name'),
        }
        # Add per-game expected values (for transparency)
        for stat in SKATER_STATS:
            out_row[f"xPG_{stat}"] = per_game[stat]

        if "row" in horizons:
            for stat in SKATER_STATS:
                out_row[f"ROW_{stat}"] = (per_game[stat] or 0.0) * games_row * m_row
        if "next" in horizons:
            for stat in SKATER_STATS:
                out_row[f"NW_{stat}"] = (per_game[stat] or 0.0) * games_next * m_next
        if "ros" in horizons:
            for stat in SKATER_STATS:
                out_row[f"ROS_{stat}"] = (per_game[stat] or 0.0) * games_ros * m_ros

        rows.append(out_row)

    out_df = pd.DataFrame(rows)

    # Order columns: id cols, per-game, then horizons grouped
    id_cols = ['Player', 'Team', 'Position', 'team_name']
    xpg_cols = [f"xPG_{s}" for s in SKATER_STATS]
    horizon_cols: List[str] = []
    if "row" in horizons:
        horizon_cols += [f"ROW_{s}" for s in SKATER_STATS]
    if "next" in horizons:
        horizon_cols += [f"NW_{s}" for s in SKATER_STATS]
    if "ros" in horizons:
        horizon_cols += [f"ROS_{s}" for s in SKATER_STATS]

    cols = [c for c in id_cols if c in out_df.columns] + xpg_cols + horizon_cols
    out_df = out_df[cols]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_csv
