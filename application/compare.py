import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SKATER_STATS = ["G", "A", "PPP", "SOG", "FOW", "HIT", "BLK", "PIM"]


def _normalize_name(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().lower()
    # remove simple punctuation commonly found in names
    for ch in [".", ",", "'", "\"", "`"]:
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return s


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
        return f if f <= 1.0 else f / 100.0
    except Exception:
        return np.nan


def _lookup_games(lookup: pd.DataFrame, team: str, current_week: int) -> Tuple[int, int, int]:
    team_rows = lookup[lookup['TM'] == team]
    if team_rows.empty:
        return 0, 0, 0
    cur = team_rows[team_rows['Week'] == current_week]
    nxt = team_rows[team_rows['Week'] == current_week + 1]
    games_row = int(cur['GamesRestOfWeek'].iloc[0]) if not cur.empty else 0
    games_ros = int(cur['GamesROS'].iloc[0]) if not cur.empty else 0
    games_next = int(nxt['Games'].iloc[0]) if not nxt.empty else 0
    return games_row, games_next, games_ros


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _load_forecasts(forecast_csv: str) -> pd.DataFrame:
    df = pd.read_csv(forecast_csv)
    # Expect columns: Player, Team, Position, team_name, xPG_*, ROW_*, NW_*, ROS_*
    _ensure_columns(df, ["Player", "Team"], "forecasts")
    # Normalize player name for joins
    df["_key_player"] = df["Player"].apply(_normalize_name)
    return df


def _load_lookup(lookup_csv: str) -> pd.DataFrame:
    lu = pd.read_csv(lookup_csv)
    _ensure_columns(lu, ["TM", "Week", "Games", "GamesRestOfWeek", "GamesROS"], "lookup_table.csv")
    return lu


def _detect_player_column(df: pd.DataFrame) -> str:
    for c in ["Player", "player", "Name", "NAME"]:
        if c in df.columns:
            return c
    raise ValueError("Projection/LY input does not contain a recognizable Player column")


def _load_natepts_projection(xlsx_path: str, sheet: str = "NatePts") -> pd.DataFrame:
    try:
        proj = pd.read_excel(xlsx_path, sheet_name=sheet)
    except Exception as e:
        raise RuntimeError(f"Failed to read projections from {xlsx_path} sheet {sheet}: {e}")

    player_col = _detect_player_column(proj)
    # try to detect team column too for better matching
    team_col = None
    for c in ["Team", "TEAM", "Tm", "tm"]:
        if c in proj.columns:
            team_col = c
            break

    # Keep only relevant columns
    keep_cols = [player_col]
    if team_col:
        keep_cols.append(team_col)
    stats_present = [s for s in SKATER_STATS if s in proj.columns]
    # FOW may be absent; that's fine
    if not stats_present:
        raise ValueError("NatePts sheet does not contain any of the expected stat columns: " + ", ".join(SKATER_STATS))

    use = proj[keep_cols + stats_present].copy()
    use.rename(columns={player_col: "Player", team_col or "Player": team_col or "Player"}, inplace=True)
    # Normalize
    use["_key_player"] = use["Player"].apply(_normalize_name)
    if team_col:
        use["_key_team"] = use[team_col].astype(str).str.strip().str.upper()
    else:
        use["_key_team"] = ""

    # These values are season totals for 82 games; convert to per-game
    for s in stats_present:
        use[f"PG_{s}"] = use[s].astype(float) / 82.0
    return use, stats_present


def _scale_pg_to_horizons(pg: float, games_row: int, games_next: int, games_ros: int) -> Tuple[float, float, float]:
    if pd.isna(pg):
        return (np.nan, np.nan, np.nan)
    return (pg * games_row, pg * games_next, pg * games_ros)


def _build_last_year_from_per60(
    sit_s_df: Optional[pd.DataFrame],
    sit_pp_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    Expect per-60 rates with TOI information. We attempt to read columns:
      - Player (name)
      - For all-situations (sit_s_df): columns for G,A,SOG,FOW,HIT,BLK,PIM per 60 and either TOI/GP or TOI, GP
      - For PP (sit_pp_df): PPP per 60 and TOI/GP (or TOI + GP)
    Column names are not strictly specified, so we try flexible matching.
    Return a dataframe with columns: Player, _key_player, PG_<STAT> for all SKATER_STATS
    """
    if sit_s_df is None and sit_pp_df is None:
        return None

    def _toi_per_game(df: pd.DataFrame) -> pd.Series:
        # try columns in order
        for cand in ["TOI/GP", "TOI_per_GP", "TOI_per_game", "TOI per GP", "TOI per game"]:
            if cand in df.columns:
                return pd.to_numeric(df[cand], errors='coerce')
        # fallback: TOI and GP
        toi_col = None
        for c in ["TOI", "toi"]:
            if c in df.columns:
                toi_col = c
                break
        gp_col = None
        for c in ["GP", "gp"]:
            if c in df.columns:
                gp_col = c
                break
        if toi_col and gp_col:
            # TOI in minutes?
            return pd.to_numeric(df[toi_col], errors='coerce') / pd.to_numeric(df[gp_col].replace(0, np.nan), errors='coerce')
        return pd.Series(np.nan, index=df.index)

    ly = pd.DataFrame()
    if sit_s_df is not None:
        s = sit_s_df.copy()
        s_player = _detect_player_column(s)
        s.rename(columns={s_player: "Player"}, inplace=True)
        s["_key_player"] = s["Player"].apply(_normalize_name)
        toi_pg_all = _toi_per_game(s)
        # map possible per60 column names to canon stat keys
        per60_map: Dict[str, str] = {}
        for stat in ["G", "A", "SOG", "FOW", "HIT", "BLK", "PIM"]:
            # candidates like G/60, G60, G_per60
            for cand in [f"{stat}/60", f"{stat}60", f"{stat}_per60", stat]:
                if cand in s.columns:
                    per60_map[stat] = cand
                    break
        for stat, col in per60_map.items():
            ly_stat = s[["_key_player"]].copy()
            ly_stat[f"PG_{stat}"] = pd.to_numeric(s[col], errors='coerce') * (pd.to_numeric(toi_pg_all, errors='coerce') / 60.0)
            if ly.empty:
                ly = ly_stat
            else:
                ly = ly.merge(ly_stat, on="_key_player", how="outer")

    if sit_pp_df is not None:
        p = sit_pp_df.copy()
        p_player = _detect_player_column(p)
        p.rename(columns={p_player: "Player"}, inplace=True)
        p["_key_player"] = p["Player"].apply(_normalize_name)
        # TOI/GP on PP
        def _toi_pp(df):
            for cand in ["TOI/GP", "PP TOI/GP", "TOI/GP_pp", "PP_TOI_per_GP"]:
                if cand in df.columns:
                    return pd.to_numeric(df[cand], errors='coerce')
            # fallback: TOI + GP
            return _toi_per_game(df)

        toi_pg_pp = _toi_pp(p)
        # Detect PPP per 60
        ppp_col = None
        for cand in ["PPP/60", "PPP60", "PPP_per60", "PPP"]:
            if cand in p.columns:
                ppp_col = cand
                break
        if ppp_col is not None:
            ppp_pg = pd.to_numeric(p[ppp_col], errors='coerce') * (pd.to_numeric(toi_pg_pp, errors='coerce') / 60.0)
            if ly.empty:
                ly = p[['_key_player']].copy()
            ly = ly.merge(pd.DataFrame({'_key_player': p['_key_player'], 'PG_PPP': ppp_pg}), on="_key_player", how="outer")

    if ly.empty:
        return None
    return ly


def compare(
    *,
    current_week: int,
    forecast_csv: str = os.path.join("data", "forecasts.csv"),
    lookup_csv: str = os.path.join("data", "lookup_table.csv"),
    proj_xlsx: str = os.path.join("NatePts.xlsx"),
    proj_sheet: str = "NatePts",
    ly_sit_s_csv: Optional[str] = None,
    ly_sit_pp_csv: Optional[str] = None,
    horizons: Tuple[str, ...] = ("row", "next", "ros"),
    out_csv: str = os.path.join("data", "compare.csv"),
    all_players: bool = False,
) -> str:
    """Build a comparison CSV of Forecast vs NatePts projections vs Last-Year.

    Returns the output CSV path.
    """
    forecasts = _load_forecasts(forecast_csv)
    lookup = _load_lookup(lookup_csv)

    # Base set of players: use forecasts (already ownership-filtered by upstream module)
    base = forecasts.copy()

    # Load NatePts projections
    proj_df, proj_stats = _load_natepts_projection(proj_xlsx, sheet=proj_sheet)

    # Merge projections to base by normalized player name (and prefer same team when available)
    merged = base.merge(proj_df[["_key_player"] + [f"PG_{s}" for s in proj_stats]], on="_key_player", how="left", suffixes=("", ""))

    # Load optional last-year per60 inputs and convert to per-game
    ly_df: Optional[pd.DataFrame] = None
    if ly_sit_s_csv or ly_sit_pp_csv:
        sit_s = pd.read_csv(ly_sit_s_csv) if ly_sit_s_csv else None
        sit_pp = pd.read_csv(ly_sit_pp_csv) if ly_sit_pp_csv else None
        ly_df = _build_last_year_from_per60(sit_s, sit_pp)
        if ly_df is not None:
            merged = merged.merge(ly_df, on="_key_player", how="left")

    # Compute horizon totals for projections and last year
    # Add placeholders for horizon totals
    for stat in proj_stats:
        for tag, col_prefix in [("row", "ROW"), ("next", "NW"), ("ros", "ROS")]:
            if tag in horizons:
                merged[f"Proj_{col_prefix}_{stat}"] = np.nan
    if ly_df is not None:
        for stat in SKATER_STATS:
            for tag, col_prefix in [("row", "ROW"), ("next", "NW"), ("ros", "ROS")]:
                if tag in horizons:
                    merged[f"LY_{col_prefix}_{stat}"] = np.nan

    # Compute using team lookup games
    def _row_apply(row: pd.Series) -> pd.Series:
        team = str(row.get('Team') or '').strip().upper()
        games_row, games_next, games_ros = _lookup_games(lookup, team, int(current_week))
        # NatePts projections
        for stat in proj_stats:
            pg = row.get(f"PG_{stat}")
            r, n, rr = _scale_pg_to_horizons(pg, games_row, games_next, games_ros)
            if "row" in horizons:
                row[f"Proj_ROW_{stat}"] = r
            if "next" in horizons:
                row[f"Proj_NW_{stat}"] = n
            if "ros" in horizons:
                row[f"Proj_ROS_{stat}"] = rr
        # Last-year
        if ly_df is not None:
            for stat in SKATER_STATS:
                pg_ly = row.get(f"PG_{stat}")  # from LY merge it shares the same PG_<stat> naming
                r2, n2, rr2 = _scale_pg_to_horizons(pg_ly, games_row, games_next, games_ros)
                if "row" in horizons:
                    row[f"LY_ROW_{stat}"] = r2
                if "next" in horizons:
                    row[f"LY_NW_{stat}"] = n2
                if "ros" in horizons:
                    row[f"LY_ROS_{stat}"] = rr2
        return row

    merged = merged.apply(_row_apply, axis=1)

    # Compute deltas: Forecast minus Proj / LY when both sides exist
    def _add_delta_cols(df: pd.DataFrame) -> pd.DataFrame:
        for stat in SKATER_STATS:
            for tag, col_prefix in [("row", "ROW"), ("next", "NW"), ("ros", "ROS")]:
                if tag not in horizons:
                    continue
                fc_col = f"{col_prefix}_{stat}"
                proj_col = f"Proj_{col_prefix}_{stat}"
                ly_col = f"LY_{col_prefix}_{stat}"
                if proj_col in df.columns:
                    df[f"Δ_Fcst_vs_Proj_{col_prefix}_{stat}"] = df.get(fc_col) - df.get(proj_col)
                if ly_col in df.columns:
                    df[f"Δ_Fcst_vs_LY_{col_prefix}_{stat}"] = df.get(fc_col) - df.get(ly_col)
        return df

    merged = _add_delta_cols(merged)

    # Build output columns order
    id_cols = [c for c in ["Player", "Team", "Position", "team_name"] if c in merged.columns]
    # Keep forecast columns first
    fc_cols: List[str] = []
    for tag, col_prefix in [("row", "ROW"), ("next", "NW"), ("ros", "ROS")]:
        if tag in horizons:
            fc_cols += [f"{col_prefix}_{s}" for s in SKATER_STATS if f"{col_prefix}_{s}" in merged.columns]
    proj_cols: List[str] = []
    for tag, col_prefix in [("row", "ROW"), ("next", "NW"), ("ros", "ROS")]:
        if tag in horizons:
            proj_cols += [f"Proj_{col_prefix}_{s}" for s in SKATER_STATS if f"Proj_{col_prefix}_{s}" in merged.columns]
    ly_cols: List[str] = []
    for tag, col_prefix in [("row", "ROW"), ("next", "NW"), ("ros", "ROS")]:
        if tag in horizons:
            ly_cols += [f"LY_{col_prefix}_{s}" for s in SKATER_STATS if f"LY_{col_prefix}_{s}" in merged.columns]
    delta_cols = [c for c in merged.columns if c.startswith("Δ_")]

    out_cols = id_cols + fc_cols + proj_cols + ly_cols + delta_cols
    out_df = merged[out_cols].copy()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # Unmatched report for projections and LY (players present in proj/ly but absent in base forecasts)
    # Build sets of keys
    base_keys = set(base["_key_player"].unique())
    unmatched_rows: List[pd.Series] = []
    # Projections unmatched
    try:
        proj_only = proj_df[~proj_df["_key_player"].isin(base_keys)].copy()
        proj_only["source"] = "NatePts"
        unmatched_rows.append(proj_only)
    except Exception:
        pass
    # LY unmatched
    try:
        if ly_df is not None:
            ly_only = ly_df[~ly_df["_key_player"].isin(base_keys)].copy()
            ly_only["source"] = "LastYear"
            unmatched_rows.append(ly_only)
    except Exception:
        pass
    if unmatched_rows:
        unmatched = pd.concat(unmatched_rows, ignore_index=True)
        unmatched.to_csv(os.path.join(os.path.dirname(out_csv), "unmatched_in_compare.csv"), index=False)

    return out_csv
