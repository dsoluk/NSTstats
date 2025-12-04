import os
import math
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

from application.forecast import SKATER_STATS


def _parse_sos(val):
    try:
        if isinstance(val, str) and val.strip().endswith("%"):
            return float(val.strip().replace("%", "")) / 100.0
        return float(val)
    except Exception:
        return np.nan


def _lookup_team_week_rows(lookup: pd.DataFrame, team: str, current_week: int) -> Tuple[int, int, int]:
    team = str(team or "").strip()
    if not team:
        return 0, 0, 0
    rows = lookup[lookup['TM'] == team]
    if rows.empty:
        return 0, 0, 0
    cur = rows[rows['Week'] == current_week]
    nxt = rows[rows['Week'] == current_week + 1]
    games_row = int(cur['GamesRestOfWeek'].iloc[0]) if not cur.empty else 0
    games_next = int(nxt['Games'].iloc[0]) if not nxt.empty else 0
    games_ros = int(cur['GamesROS'].iloc[0]) if not cur.empty else 0
    return games_row, games_next, games_ros


def _safe_get(df: pd.DataFrame, row: pd.Series, col: str, default=np.nan):
    try:
        return row.get(col, default)
    except Exception:
        return default


def _is_waiver_row(row: pd.Series) -> bool:
    # Treat blank or NaN team_name as waiver wire
    tn_val = row.get('team_name') if 'team_name' in row else ''
    if pd.isna(tn_val):
        return True
    tn = str(tn_val).strip()
    return tn == ''


def _is_team_row(row: pd.Series, team_name: str) -> bool:
    tn = str(row.get('team_name') if 'team_name' in row else '').strip()
    return tn == str(team_name).strip()


def _pos_group_of(row: pd.Series) -> str:
    # Use explicit pos_group if present; otherwise infer F/D from Elig_Pos/Position
    pg = str(row.get('pos_group') if 'pos_group' in row else '').strip()
    if pg in ('F', 'D'):
        return pg
    elig = str(row.get('Elig_Pos') or row.get('Position') or '').upper()
    if 'D' in elig and ('LW' not in elig and 'RW' not in elig and 'C' not in elig):
        return 'D'
    return 'F'


def _composite_l7(row: pd.Series) -> float:
    # Prefer Composite_Index_l7 if available; otherwise average offensive/banger l7, else mean of T_l7 metrics
    if 'Composite_Index_l7' in row and not pd.isna(row['Composite_Index_l7']):
        return float(row['Composite_Index_l7'])
    oi = row.get('Offensive_Index_l7', np.nan)
    bi = row.get('Banger_Index_l7', np.nan)
    if not pd.isna(oi) and not pd.isna(bi):
        return float((oi + bi) / 2.0)
    # fallback to mean of l7 T-scores for tracked stats
    vals = []
    for s in SKATER_STATS:
        v = row.get(f"T_l7_{s}", np.nan)
        if not pd.isna(v):
            vals.append(float(v))
    return float(np.nanmean(vals)) if vals else 0.0


def _composite_szn(row: pd.Series) -> float:
    if 'Composite_Index_szn' in row and not pd.isna(row['Composite_Index_szn']):
        return float(row['Composite_Index_szn'])
    oi = row.get('Offensive_Index_szn', np.nan)
    bi = row.get('Banger_Index_szn', np.nan)
    if not pd.isna(oi) and not pd.isna(bi):
        return float((oi + bi) / 2.0)
    vals = []
    for s in SKATER_STATS:
        v = row.get(f"T_szn_{s}", np.nan)
        if not pd.isna(v):
            vals.append(float(v))
    return float(np.nanmean(vals)) if vals else 0.0


def _ros_improvement_boost(row: pd.Series, prior_map: Dict[str, float]) -> float:
    # Boost ROS if current szn composite exceeds last year composite
    player = str(row.get('Player'))
    cur = _composite_szn(row)
    ly = prior_map.get(player, np.nan)
    if pd.isna(ly):
        return 0.0
    diff = float(cur - ly)
    # Scale modestly; cap boost
    return max(0.0, min(diff, 20.0)) * 0.2  # up to +4 boost


def _collect_t_scores_block(row: pd.Series) -> Dict[str, float]:
    out = {}
    for s in SKATER_STATS:
        out[f"T_szn_{s}"] = float(row.get(f"T_szn_{s}", np.nan)) if not pd.isna(row.get(f"T_szn_{s}", np.nan)) else np.nan
        out[f"T_l7_{s}"] = float(row.get(f"T_l7_{s}", np.nan)) if not pd.isna(row.get(f"T_l7_{s}", np.nan)) else np.nan
    out['Offensive_Index_szn'] = row.get('Offensive_Index_szn', np.nan)
    out['Banger_Index_szn'] = row.get('Banger_Index_szn', np.nan)
    out['Composite_Index_szn'] = row.get('Composite_Index_szn', np.nan)
    out['Offensive_Index_l7'] = row.get('Offensive_Index_l7', np.nan)
    out['Banger_Index_l7'] = row.get('Banger_Index_l7', np.nan)
    out['Composite_Index_l7'] = row.get('Composite_Index_l7', np.nan)
    return out


def _ensure_dirs(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def recommend(
    team_name: str,
    current_week: int,
    skaters_csv: str = os.path.join("data", "merged_skaters.csv"),
    lookup_csv: str = os.path.join("data", "lookup_table.csv"),
    prior_csv: str = os.path.join("data", "merged_skaters_prior.csv"),
    out_dir: str = os.path.join("output"),
) -> Dict[str, str]:
    """Build waiver-wire streaming recommendations for This Week, Next Week, and ROS.

    Returns dict with paths to the CSV result and plot directory.
    """
    sk = pd.read_csv(skaters_csv)
    lu = pd.read_csv(lookup_csv)

    # prior map for ROS improvement
    prior_map: Dict[str, float] = {}
    if os.path.exists(prior_csv):
        try:
            prior = pd.read_csv(prior_csv)
            # Some prior files may already have Composite_Index_szn; otherwise avg T_szn_*
            comp_vals = {}
            for _, r in prior.iterrows():
                player = str(r.get('Player'))
                if 'Composite_Index_szn' in r and not pd.isna(r['Composite_Index_szn']):
                    comp_vals[player] = float(r['Composite_Index_szn'])
                else:
                    vals = []
                    for s in SKATER_STATS:
                        v = r.get(f"T_szn_{s}", np.nan)
                        if not pd.isna(v):
                            vals.append(float(v))
                    comp_vals[player] = float(np.nanmean(vals)) if vals else np.nan
            prior_map = comp_vals
        except Exception:
            prior_map = {}

    # Guarantee necessary columns
    if 'Team' not in sk.columns and 'TM' in sk.columns:
        sk.rename(columns={'TM': 'Team'}, inplace=True)

    # Partition roster vs waiver wire
    roster = sk[sk.apply(lambda r: _is_team_row(r, team_name), axis=1)].copy()
    waiver = sk[sk.apply(_is_waiver_row, axis=1)].copy()

    if roster.empty:
        raise ValueError(f"No players found for team_name='{team_name}' in {skaters_csv}")

    # Helper to compute horizon score per row
    def horizon_scores(row: pd.Series) -> Dict[str, float]:
        team = str(row.get('Team') or '').strip()
        games_row, games_next, games_ros = _lookup_team_week_rows(lu, team, int(current_week))
        comp_l7 = _composite_l7(row)
        comp_szn = _composite_szn(row)
        this_week = comp_l7 * games_row
        next_week = comp_l7 * games_next
        ros = comp_szn * games_ros + _ros_improvement_boost(row, prior_map)
        return {"this_week": this_week, "next_week": next_week, "ros": ros,
                "games_row": games_row, "games_next": games_next, "games_ros": games_ros,
                "comp_l7": comp_l7, "comp_szn": comp_szn}

    # Add computed scores
    for df in (roster, waiver):
        hs = df.apply(horizon_scores, axis=1, result_type='expand')
        for col in hs.columns:
            df[col] = hs[col]
        df['pos_group_eff'] = df.apply(_pos_group_of, axis=1)

    # Select top waiver candidates by pos and horizon
    def top_candidates(df: pd.DataFrame, horizon: str, pos: str, k: int = 3) -> pd.DataFrame:
        tdf = df[df['pos_group_eff'] == pos].copy()
        if tdf.empty:
            return tdf
        tdf.sort_values(by=[horizon, 'comp_l7', 'comp_szn'], ascending=[False, False, False], inplace=True)
        return tdf.head(k)

    def worst_roster(df: pd.DataFrame, horizon: str, pos: str, k: int = 3) -> pd.DataFrame:
        tdf = df[df['pos_group_eff'] == pos].copy()
        if tdf.empty:
            return tdf
        tdf.sort_values(by=[horizon, 'comp_l7', 'comp_szn'], ascending=[True, True, True], inplace=True)
        return tdf.head(k)

    horizons = ["this_week", "next_week", "ros"]
    suggestions: List[Dict[str, object]] = []

    for horizon in horizons:
        for pos in ("F", "D"):
            adds = top_candidates(waiver, horizon, pos, k=3)
            drops = worst_roster(roster, horizon, pos, k=3)
            used_drops: set = set()
            for _, add_row in adds.iterrows():
                # Find the drop with strictly lower score than add
                best_choice = None
                best_delta = 0.0
                for _, dr in drops.iterrows():
                    if dr['Player'] in used_drops:
                        continue
                    delta = float(add_row[horizon]) - float(dr[horizon])
                    if delta > best_delta:
                        best_delta = delta
                        best_choice = dr
                if best_choice is not None and best_delta > 0.0:
                    used_drops.add(best_choice['Player'])
                    block_add = _collect_t_scores_block(add_row)
                    block_drop = _collect_t_scores_block(best_choice)
                    suggestions.append({
                        "scenario": horizon,
                        "pos_group": pos,
                        "add_player": add_row['Player'],
                        "add_team": add_row.get('Team'),
                        "add_games": int(add_row['games_row'] if horizon == 'this_week' else add_row['games_next'] if horizon == 'next_week' else add_row['games_ros']),
                        "drop_player": best_choice['Player'],
                        "drop_team": best_choice.get('Team'),
                        "drop_games": int(best_choice['games_row'] if horizon == 'this_week' else best_choice['games_next'] if horizon == 'next_week' else best_choice['games_ros']),
                        "proj_improvement": round(best_delta, 2),
                        **{f"add_{k}": v for k, v in block_add.items()},
                        **{f"drop_{k}": v for k, v in block_drop.items()},
                    })

    recs = pd.DataFrame(suggestions)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_dir or "output"
    plots_dir = os.path.join("_plots", "waiver_agent", timestamp)
    _ensure_dirs(out_dir)
    _ensure_dirs(plots_dir)

    csv_path = os.path.join(out_dir, f"waiver_recs_{timestamp}.csv")
    recs.to_csv(csv_path, index=False)

    # Create bar plots
    try:
        import matplotlib.pyplot as plt
        # One figure per scenario/pos showing top deltas
        for horizon in horizons:
            for pos in ("F", "D"):
                sub = recs[(recs['scenario'] == horizon) & (recs['pos_group'] == pos)].copy()
                if sub.empty:
                    continue
                # Plot pairs side-by-side for add vs drop expected horizon score
                labels = [f"{r.add_player} → {r.drop_player}" for _, r in sub.iterrows()]
                add_vals = []
                drop_vals = []
                for _, r in sub.iterrows():
                    # derive horizon scores from the original dataframes
                    # We included proj_improvement but not absolute values; recompute from comp/games
                    add_games = r['add_games']
                    drop_games = r['drop_games']
                    add_comp = r.get('add_Composite_Index_l7') if horizon != 'ros' else r.get('add_Composite_Index_szn')
                    drop_comp = r.get('drop_Composite_Index_l7') if horizon != 'ros' else r.get('drop_Composite_Index_szn')
                    if pd.isna(add_comp):
                        add_comp = float(np.nanmean([r.get(f"add_T_l7_{s}") for s in SKATER_STATS])) if horizon != 'ros' else float(np.nanmean([r.get(f"add_T_szn_{s}") for s in SKATER_STATS]))
                    if pd.isna(drop_comp):
                        drop_comp = float(np.nanmean([r.get(f"drop_T_l7_{s}") for s in SKATER_STATS])) if horizon != 'ros' else float(np.nanmean([r.get(f"drop_T_szn_{s}") for s in SKATER_STATS]))
                    add_vals.append(float(add_comp) * float(add_games))
                    drop_vals.append(float(drop_comp) * float(drop_games))

                x = np.arange(len(labels))
                width = 0.38
                fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.6), 4))
                ax.bar(x - width/2, drop_vals, width, label='Drop (roster)')
                ax.bar(x + width/2, add_vals, width, label='Add (waiver)')
                ax.set_ylabel('Expected Composite x Games')
                title_map = {"this_week": "This Week Boost", "next_week": "Next Week Streamers", "ros": "Rest of Season Impact"}
                ax.set_title(f"{title_map.get(horizon, horizon)} – {pos}")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=20, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                fig.tight_layout()
                plot_path = os.path.join(plots_dir, f"{horizon}_{pos}.png")
                fig.savefig(plot_path, dpi=160)
                plt.close(fig)
    except Exception:
        # plotting is optional; ignore errors in headless or missing matplotlib
        pass

    return {"csv": csv_path, "plots_dir": plots_dir}
