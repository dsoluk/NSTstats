import os
from typing import Tuple

import pandas as pd

from registries.names import NamesRegistry
from registries.normalize import normalize_name
from registries.positions import PositionRegistry
from registries.teams import TeamRegistry


def _prepare_registry_df() -> pd.DataFrame:
    names = NamesRegistry()
    names.load()
    rows = []
    for r in names.all():
        rows.append({
            'registry_id': r.id,
            'cname': r.cname,
            'reg_team': r.team,
            'reg_pos': r.position,
            'reg_display_name': r.display_name,
            'reg_source': r.source,
        })
    return pd.DataFrame(rows)


def _normalize_team_pos(team_registry: TeamRegistry, pos_registry: PositionRegistry, team: str, pos: str) -> Tuple[str, str]:
    t = team_registry.to_code(team)
    p = pos_registry.normalize(pos)
    return t or '', p or ''


def _load_yahoo_all_rosters(csv_path: str, pos_registry: PositionRegistry) -> pd.DataFrame:
    """
    Load Yahoo all_rosters.csv and return per-player rows with:
      - cname: normalized name for joining
      - ypos: normalized primary position for joining (C/L/R/D/G)
      - yahoo_name: the Yahoo-displayed player name
      - yahoo_positions: semicolon-joined eligible positions from Yahoo, with 'Util' removed
      - team_name: Yahoo fantasy team name that currently rosters the player
    Joining elsewhere will be on (cname, ypos); team is intentionally ignored for the match.
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['cname', 'ypos', 'yahoo_name', 'yahoo_positions', 'team_name'])
    ydf = pd.read_csv(csv_path)
    # Normalize name to cname
    ydf['cname'] = ydf['name'].fillna('').apply(normalize_name)

    def map_pos(row) -> str:
        sel = str(row.get('selected_position') or '').upper()
        poss = str(row.get('positions') or '')
        # positions is ";"-separated
        parts = [p.strip().upper() for p in poss.split(';') if p.strip()]
        # Prefer selected_position if available and not bench/IR
        pref = sel if sel and sel not in {'BN', 'IR', 'IR+', 'NA'} else ''
        order = ([pref] if pref else []) + parts
        # Map to our codes using PositionRegistry
        for p in order:
            code = pos_registry.normalize(p)
            if code in {'G', 'D', 'C', 'L', 'R'}:
                return code
        # Fallbacks
        if 'D' in parts:
            return 'D'
        if 'G' in parts:
            return 'G'
        # Any forward variants → try to coerce to C by default
        for cand in ['C', 'LW', 'RW']:
            code = pos_registry.normalize(cand)
            if code:
                return code
        return ''

    # Primary normalized position for joining
    ydf['ypos'] = ydf.apply(map_pos, axis=1)

    # Build yahoo_positions (eligible set) with 'UTIL' removed, preserve original Yahoo labels
    def build_yahoo_positions(poss: str) -> str:
        parts = [p.strip() for p in str(poss).split(';') if p.strip()]
        # Remove Util/UTIL case-insensitively
        parts = [p for p in parts if p.upper() != 'UTIL']
        # Return semicolon-joined string
        return ';'.join(parts)

    ydf['yahoo_positions'] = ydf['positions'].apply(build_yahoo_positions)
    ydf['yahoo_name'] = ydf['name']

    # Reduce to one row per (cname, ypos) by picking any non-empty yahoo_name and union of eligible positions
    if not ydf.empty:
        # Aggregate eligible positions by union (set) while keeping a representative name (first non-null)
        def agg_positions(series: pd.Series) -> str:
            acc = []
            seen = set()
            for s in series.dropna().astype(str):
                for p in s.split(';') if s else []:
                    if p and p not in seen:
                        seen.add(p); acc.append(p)
            return ';'.join(acc)

        def first_non_null(series: pd.Series):
            for v in series:
                if pd.notna(v) and str(v).strip() != '':
                    return v
            return None

        agg = (
            ydf.dropna(subset=['cname', 'ypos'])
               .groupby(['cname', 'ypos'], as_index=False)
               .agg(yahoo_name=('yahoo_name', first_non_null),
                    yahoo_positions=('yahoo_positions', agg_positions),
                    team_name=('team_name', first_non_null))
        )
        return agg

    return pd.DataFrame(columns=['cname', 'ypos', 'yahoo_name', 'yahoo_positions', 'team_name'])


def merge_role(nst_csv: str, role: str, out_csv: str) -> str:
    """
    Merge NST stats with Player Registry, then with Yahoo ownership.
    role: 'skater' or 'goalie'
    Returns output CSV path.
    """
    team_registry = TeamRegistry()
    pos_registry = PositionRegistry()
    team_registry.load()

    # Load NST
    if not os.path.exists(nst_csv):
        raise FileNotFoundError(f"NST CSV not found: {nst_csv}")
    df = pd.read_csv(nst_csv)

    # Prepare NST keys
    df['cname'] = df['Player'].astype(str).apply(normalize_name)
    if role == 'goalie':
        df['Position'] = 'G'
    df['pos_code'] = df['Position'].astype(str).apply(pos_registry.normalize)
    df['team_code'] = df['Team'].astype(str).apply(team_registry.to_code)

    # Load Registry
    reg = _prepare_registry_df()

    # Merge NST with registry on cname+team+pos
    merged = df.merge(
        reg,
        left_on=['cname', 'team_code', 'pos_code'],
        right_on=['cname', 'reg_team', 'reg_pos'],
        how='left'
    )

    # Load Yahoo roster info and merge on cname+position (no team)
    yinfo = _load_yahoo_all_rosters(os.path.join('data', 'all_rosters.csv'), pos_registry)
    if not yinfo.empty:
        merged = merged.merge(
            yinfo,
            left_on=['cname', 'pos_code'],
            right_on=['cname', 'ypos'],
            how='left'
        )
        # Remove merge helper
        if 'ypos' in merged.columns:
            merged.drop(columns=['ypos'], inplace=True)
    else:
        # Ensure yahoo columns exist even if empty
        merged['yahoo_name'] = None
        merged['yahoo_positions'] = None
        merged['team_name'] = None

    # Create Elig_Pos from yahoo_positions (and keep even if empty)
    if 'yahoo_positions' in merged.columns:
        merged['Elig_Pos'] = merged['yahoo_positions']
    else:
        merged['Elig_Pos'] = None

    # Drop fields per request (updated)
    drop_cols = [
        'registry_id', 'reg_display_name', 'cname', 'team_code', 'pos_code',
        'reg_team', 'reg_pos', 'reg_source', 'owned_count', 'owned_pct',
        'Position', 'yahoo_name', 'yahoo_positions'
    ]
    existing_drops = [c for c in drop_cols if c in merged.columns]
    if existing_drops:
        merged.drop(columns=existing_drops, inplace=True)

    # Reorder to surface key fields near front
    front = [c for c in ['Player', 'Team', 'team_name', 'Elig_Pos'] if c in merged.columns]
    others = [c for c in merged.columns if c not in front]
    merged = merged[front + others]

    # Special-case note: Elias Pettersson disambiguation happens via position in the join above.

    # Ensure output dir
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return out_csv


def run_merge():
    out1 = merge_role(os.path.join('data', 'skaters.csv'), 'skater', os.path.join('data', 'merged_skaters.csv'))
    out2 = merge_role(os.path.join('data', 'goalies.csv'), 'goalie', os.path.join('data', 'merged_goalies.csv'))
    print(f"Saved merged skaters to {out1}")
    print(f"Saved merged goalies to {out2}")
