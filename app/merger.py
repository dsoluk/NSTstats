import os
from typing import Optional

import pandas as pd

from helpers.normalization import normalize_name, normalize_position, to_team_code, to_fpos
from infrastructure.persistence import get_session, CurrentRoster, Player, Team, League
from sqlalchemy.orm import joinedload






def _load_yahoo_rosters_from_db(league_id: Optional[int] = None) -> pd.DataFrame:
    """
    Load roster mapping from database (CurrentRoster join Player join Team).
    """
    session = get_session()
    try:
        # We need to know which league.
        # If league_id is provided, filter by it.
        q = (
            session.query(
                Player.player_key,
                Player.name,
                Player.positions,
                Player.status,
                Team.team_name,
                CurrentRoster.team_key
            )
            .join(CurrentRoster, Player.player_key == CurrentRoster.player_key)
            .join(Team, CurrentRoster.team_key == Team.team_key)
        )
        if league_id is not None:
            q = q.filter(Team.league_id == league_id)
            
        rows = []
        for p_key, p_name, p_pos, p_status, t_name, t_key in q.all():
            rows.append({
                'player_key': p_key,
                'name': p_name,
                'positions': p_pos,
                'status': p_status,
                'team_name': t_name,
                'team_key': t_key
            })
        
        # Also include unowned players that have a status (IR, IR+, etc.)
        # For a specific league merge, we might want to know who is IR in THAT league.
        # But Player.status is global currently. 
        # TODO: Yahoo rosters often have different status per league (e.g. NA in one, IR in another).
        unowned_q = (
            session.query(Player.player_key, Player.name, Player.positions, Player.status)
            .filter(Player.status.isnot(None))
        )
        if league_id is not None:
            # Filter to players not in CurrentRoster for THIS league
            unowned_q = unowned_q.filter(~Player.player_key.in_(
                session.query(CurrentRoster.player_key).filter(CurrentRoster.league_id == league_id)
            ))
        else:
            unowned_q = unowned_q.filter(~Player.player_key.in_(session.query(CurrentRoster.player_key)))

        for p_key, p_name, p_pos, p_status in unowned_q.all():
            rows.append({
                'player_key': p_key,
                'name': p_name,
                'positions': p_pos,
                'status': p_status,
                'team_name': None,
                'team_key': None
            })

        ydf = pd.DataFrame(rows)
        if ydf.empty:
            return pd.DataFrame(columns=['cname', 'ypos', 'fpos', 'yahoo_name', 'yahoo_positions', 'team_name', 'status'])
        
        # Normalize as before
        ydf['cname'] = ydf['name'].fillna('').apply(normalize_name)
        
        def map_pos(row) -> str:
            poss = str(row.get('positions') or '')
            parts = [p.strip().upper() for p in poss.split(';') if p.strip()]
            for p in parts:
                code = normalize_position(p)
                if code in {'G', 'D', 'C', 'L', 'R'}:
                    return code
            if 'D' in parts: return 'D'
            if 'G' in parts: return 'G'
            return ''

        ydf['ypos'] = ydf.apply(map_pos, axis=1)
        ydf['fpos'] = ydf['ypos'].apply(to_fpos)
        
        def build_yahoo_positions(poss: str) -> str:
            parts = [p.strip() for p in str(poss).split(';') if p.strip()]
            parts = [p for p in parts if p.upper() != 'UTIL']
            return ';'.join(parts)

        ydf['yahoo_positions'] = ydf['positions'].apply(build_yahoo_positions)
        ydf['yahoo_name'] = ydf['name']

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
               .groupby(['player_key', 'cname', 'ypos', 'fpos'], as_index=False)
               .agg(yahoo_name=('yahoo_name', first_non_null),
                    yahoo_positions=('yahoo_positions', agg_positions),
                    team_name=('team_name', first_non_null),
                    status=('status', first_non_null))
        )
        return agg
    finally:
        session.close()

def _load_yahoo_all_rosters(csv_path: str, league_id: Optional[int] = None) -> pd.DataFrame:
    """
    Attempt to load from DB first, fall back to CSV if DB is empty.
    """
    df = _load_yahoo_rosters_from_db(league_id)
    if not df.empty:
        if league_id is not None:
            print(f"[Info] Loaded roster mapping from database for league_id={league_id}.")
        else:
            print("[Info] Loaded roster mapping from database.")
        return df
    
    if league_id is not None:
        print(f"[Info] Database rosters empty for league_id={league_id}, falling back to {csv_path}")
    else:
        print(f"[Info] Database rosters empty, falling back to {csv_path}")
    
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['cname', 'ypos', 'fpos', 'yahoo_name', 'yahoo_positions', 'team_name', 'status'])
    ydf = pd.read_csv(csv_path)
    
    # Filter CSV by league_id if it has it
    if league_id is not None and 'league_id' in ydf.columns:
        ydf = ydf[ydf['league_id'] == league_id].copy()
    
    if 'status' not in ydf.columns:
        ydf['status'] = None
    # Normalize name to cname
    ydf['cname'] = ydf['name'].fillna('').apply(normalize_name)
    if 'player_key' not in ydf.columns:
        ydf['player_key'] = None

    def map_pos(row) -> str:
        sel = str(row.get('selected_position') or '').upper()
        poss = str(row.get('positions') or '')
        # positions is ";"-separated
        parts = [p.strip().upper() for p in poss.split(';') if p.strip()]
        # Prefer selected_position if available and not bench/IR
        pref = sel if sel and sel not in {'BN', 'IR', 'IR+', 'NA'} else ''
        order = ([pref] if pref else []) + parts
        # Map to our codes using helpers.normalize_position
        for p in order:
            code = normalize_position(p)
            if code in {'G', 'D', 'C', 'L', 'R'}:
                return code
        # Fallbacks
        if 'D' in parts:
            return 'D'
        if 'G' in parts:
            return 'G'
        # Any forward variants → try to coerce to C by default
        for cand in ['C', 'LW', 'RW']:
            code = normalize_position(cand)
            if code:
                return code
        return ''

    # Primary normalized position for joining
    ydf['ypos'] = ydf.apply(map_pos, axis=1)
    # Coarse position for fallback
    ydf['fpos'] = ydf['ypos'].apply(to_fpos)

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
               .groupby(['player_key', 'cname', 'ypos', 'fpos'], as_index=False)
               .agg(yahoo_name=('yahoo_name', first_non_null),
                    yahoo_positions=('yahoo_positions', agg_positions),
                    team_name=('team_name', first_non_null),
                    status=('status', first_non_null))
        )
        return agg

    return pd.DataFrame(columns=['player_key', 'cname', 'ypos', 'fpos', 'yahoo_name', 'yahoo_positions', 'team_name'])


def merge_role(nst_csv: str, role: str, out_csv: str, league_id: Optional[int] = None) -> str:
    """
    Merge stats with Yahoo ownership using local normalization (no registry).
    role: 'skater' or 'goalie'
    Returns output CSV path.
    """

    # Load stats
    if not os.path.exists(nst_csv):
        raise FileNotFoundError(f"Stats CSV not found: {nst_csv}")
    df = pd.read_csv(nst_csv)

    # Ensure TOI/GP columns are MM:SS for Excel (skaters only)
    if role == 'skater':
        try:
            import pandas as _pd
            import numpy as _np

            def _fmt_hms_from_seconds(sec_val):
                if _pd.isna(sec_val):
                    return ""
                try:
                    sec = float(sec_val)
                except Exception:
                    return ""
                if _np.isnan(sec):
                    return ""
                neg = sec < 0
                # Truncate toward zero then format absolute value
                sec = abs(int(sec))
                hours = sec // 3600
                minutes = (sec % 3600) // 60
                seconds = sec % 60
                out = f"{hours}:{minutes:02d}:{seconds:02d}"
                return f"-{out}" if neg else out

            def _coerce_to_seconds(series: _pd.Series) -> _pd.Series:
                # Try timedelta first (handles '0 days 00:18:12' and '00:18:12')
                td = _pd.to_timedelta(series, errors='coerce')
                if td.notna().any():
                    return td.dt.total_seconds()
                # If already in M:SS or MM:SS or numeric, try to parse manually
                s = series.astype(str)
                # Pattern like 12:34 or 1:02
                mask = s.str.match(r"^\s*\d{1,3}:\d{2}\s*$")
                sec = _pd.Series(_np.nan, index=series.index, dtype='float64')
                if mask.any():
                    parts = s[mask].str.strip().str.split(':', n=1, expand=True)
                    sec.loc[mask] = parts[0].astype(int) * 60 + parts[1].astype(int)
                # Try raw numeric seconds
                num_mask = ~mask
                with _pd.option_context('mode.use_inf_as_na', True):
                    sec.loc[num_mask] = _pd.to_numeric(s[num_mask], errors='coerce')
                return sec

            toi_cols = [c for c in df.columns if 'TOI/GP' in str(c)]
            for c in toi_cols:
                secs = _coerce_to_seconds(df[c])
                if secs.notna().any():
                    df[c] = secs.apply(_fmt_hms_from_seconds)
        except Exception as _e:
            # Non-fatal; leave as-is if something unexpected happens
            print(f"[Warn] TOI/GP formatting skipped due to error: {_e}")

    # Prepare keys
    df['cname'] = df['Player'].astype(str).apply(normalize_name)
    if role == 'goalie':
        df['Position'] = 'G'
    # Normalize team/position using local helpers
    def _last_token(val: str) -> str:
        s = str(val or '').strip()
        if ',' in s:
            parts = [p.strip() for p in s.split(',') if p.strip()]
            if parts:
                return parts[-1]
        return s
    # If role goalie, force G; else use provided Position and select last if comma-separated
    if role != 'goalie':
        df['Position'] = df['Position'].astype(str).apply(_last_token)
    df['pos_code'] = df['Position'].astype(str).apply(normalize_position)
    df['team_code'] = df['Team'].astype(str).apply(_last_token).apply(to_team_code)

    # Start merged from stats; no registry
    merged = df.copy()

    # Load Yahoo roster info and merge on cname+position (no team)
    yinfo = _load_yahoo_all_rosters(os.path.join('data', 'all_rosters.csv'), league_id)
    if not yinfo.empty:
        # If we have player_key in our stats (Yahoo sourced), join on it first
        if 'player_key' in merged.columns and 'player_key' in yinfo.columns:
            # Ensure dtypes match
            merged['player_key'] = merged['player_key'].astype(str)
            yinfo['player_key'] = yinfo['player_key'].astype(str)
            
            merged = merged.merge(
                yinfo[['player_key', 'team_name', 'yahoo_positions', 'yahoo_name', 'status']],
                on='player_key',
                how='left'
            )
            # Fill missing via cname/pos below if any pkeys didn't match
        
        if 'yahoo_name' not in merged.columns or merged['yahoo_name'].isna().any():
            # Exact match on (cname, exact position)
            merged = merged.merge(
                yinfo[['cname', 'ypos', 'team_name', 'yahoo_positions', 'yahoo_name', 'status']],
                left_on=['cname', 'pos_code'],
                right_on=['cname', 'ypos'],
                how='left',
                suffixes=('', '_p')
            )
            # Resolve suffixes
            for col in ['team_name', 'yahoo_positions', 'yahoo_name', 'status']:
                pcol = f"{col}_p"
                if pcol in merged.columns:
                    merged[col] = merged[col].fillna(merged[pcol])
                    merged.drop(columns=[pcol], inplace=True)
            if 'ypos' in merged.columns:
                merged.drop(columns=['ypos'], inplace=True)

        # Coarse F/D/G fallback for rows still missing team_name
        # Coarse F/D/G fallback for rows still missing team_name
        def _to_fpos(p: str) -> str:
            p = str(p).upper()
            if p in {'C', 'L', 'R'}:
                return 'F'
            if p in {'D', 'G'}:
                return p
            return ''
        merged['fpos'] = merged['pos_code'].apply(_to_fpos)
        need_fallback = merged['yahoo_name'].isna()
        if need_fallback.any():
            y_fg = yinfo[['cname', 'fpos', 'team_name', 'yahoo_positions', 'yahoo_name', 'status']].copy()
            y_fg = y_fg.dropna(subset=['fpos'])
            merged = merged.merge(
                y_fg,
                left_on=['cname', 'fpos'],
                right_on=['cname', 'fpos'],
                how='left',
                suffixes=('', '_fg')
            )
            for col in ['team_name', 'yahoo_positions', 'yahoo_name', 'status']:
                alt = f"{col}_fg"
                if alt in merged.columns:
                    merged[col] = merged[col].fillna(merged[alt])
            drop_alt = [c for c in ['team_name_fg', 'yahoo_positions_fg', 'yahoo_name_fg', 'status_fg'] if c in merged.columns]
            if drop_alt:
                merged.drop(columns=drop_alt, inplace=True)
        # Final fallback: match on cname only (may be ambiguous for duplicate names)
        need_final = merged['yahoo_name'].isna()
        if need_final.any():
            y_by_name = (
                yinfo.groupby('cname', as_index=False)
                     .agg(team_name=('team_name', 'first'),
                          yahoo_positions=('yahoo_positions', 'first'),
                          yahoo_name=('yahoo_name', 'first'),
                          status=('status', 'first'))
            )
            merged = merged.merge(
                y_by_name,
                on='cname',
                how='left',
                suffixes=('', '_byname')
            )
            for col in ['team_name', 'yahoo_positions', 'yahoo_name', 'status']:
                alt = f"{col}_byname"
                if alt in merged.columns:
                    merged[col] = merged[col].fillna(merged[alt])
            drop_alt = [c for c in ['team_name_byname', 'yahoo_positions_byname', 'yahoo_name_byname', 'status_byname'] if c in merged.columns]
            if drop_alt:
                merged.drop(columns=drop_alt, inplace=True)
        # Remove merge helpers
        for helper in ['ypos', 'fpos']:
            if helper in merged.columns:
                merged.drop(columns=[helper], inplace=True)
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
        'cname', 'team_code', 'pos_code',
        'owned_count', 'owned_pct',
        'Position', 'yahoo_name', 'yahoo_positions'
    ]
    existing_drops = [c for c in drop_cols if c in merged.columns]
    if existing_drops:
        merged.drop(columns=existing_drops, inplace=True)

    # Reorder to surface key fields near front
    front = [c for c in ['Player', 'Team', 'team_name', 'Elig_Pos', 'status'] if c in merged.columns]
    others = [c for c in merged.columns if c not in front]
    merged = merged[front + others]

    # Special-case note: Elias Pettersson disambiguation happens via position in the join above.

    # Ensure output dir
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return out_csv


def run_merge():
    session = get_session()
    try:
        leagues = session.query(League).all()
        if not leagues:
            print("[Warn] No leagues found in DB. Performing default merge.")
            leagues = [None]
        
        for league in leagues:
            league_id_str = ""
            league_id = None
            if league:
                league_id = league.id
                league_id_str = league.league_key.split(".l.")[-1]
            
            suffix = f"_{league_id_str}" if league_id_str else ""
            
            # For each league, we use the scored file for THAT league if it exists, otherwise fall back to default
            sk_scored = os.path.join('data', f'skaters_scored{suffix}.csv')
            if not os.path.exists(sk_scored): sk_scored = os.path.join('data', 'skaters_scored.csv')
            
            g_scored = os.path.join('data', f'goalies_scored{suffix}.csv')
            if not os.path.exists(g_scored): g_scored = os.path.join('data', 'goalies_scored.csv')

            out1 = merge_role(sk_scored, 'skater', os.path.join('data', f'merged_skaters{suffix}.csv'), league_id)
            out2 = merge_role(g_scored, 'goalie', os.path.join('data', f'merged_goalies{suffix}.csv'), league_id)
            
            # For backward compatibility, also save to default path if it's the first/only league
            if league == leagues[0]:
                import shutil
                shutil.copy(out1, os.path.join('data', 'merged_skaters.csv'))
                shutil.copy(out2, os.path.join('data', 'merged_goalies.csv'))

            print(f"Saved merged skaters for {league.league_key if league else 'default'} to {out1}")
            print(f"Saved merged goalies for {league.league_key if league else 'default'} to {out2}")

            # DQ report per league? For now just keep the default behavior for the first league
            if league == leagues[0]:
                _run_dq_report(out1, out2, league_id)

    finally:
        session.close()

def _run_dq_report(out1, out2, league_id=None):
    # Lightweight data-quality report for merge coverage
    try:
        import json as _json
        
        # Load Yahoo rosters for this league
        yinfo = _load_yahoo_all_rosters(os.path.join('data', 'all_rosters.csv'), league_id)
        if yinfo.empty:
            print(f"[Info] No Yahoo rosters found for league_id={league_id}; skipping DQ report.")
            return

        # Load merged outputs without converting "None" or empty strings to NaN
        ms = pd.read_csv(out1, na_filter=False)
        mg = pd.read_csv(out2, na_filter=False)
        
        # Count matched players with a team
        sk_with_team = int((ms['team_name'] != '').sum()) if 'team_name' in ms.columns else 0
        g_with_team = int((mg['team_name'] != '').sum()) if 'team_name' in mg.columns else 0
        total_with_team = sk_with_team + g_with_team

        # Identify matched names (including unowned)
        matched_names = set()
        for df in [ms, mg]:
            if not df.empty and 'Elig_Pos' in df.columns:
                # If matched, Elig_Pos will be non-empty (even if it's "None")
                matched_names.update(df.loc[df['Elig_Pos'] != '', 'Player'].astype(str).apply(normalize_name))

        # Identify missing Yahoo players
        missing = yinfo[~yinfo['cname'].isin(matched_names)].copy()
        
        # Identify all NST players for the reason field
        nst_names = set()
        if not ms.empty: nst_names.update(ms['Player'].astype(str).apply(normalize_name))
        if not mg.empty: nst_names.update(mg['Player'].astype(str).apply(normalize_name))

        if not missing.empty:
            # Rename columns for clarity in the output CSV
            missing.rename(columns={'yahoo_name': 'name', 'yahoo_positions': 'positions'}, inplace=True)
            
            def get_reason(row):
                if row['cname'] not in nst_names:
                    return "Missing from NST (Injured/AHL?)"
                return "Position match failed"

            missing['reason'] = missing.apply(get_reason, axis=1)
            
            # Save unmatched report
            out_missing = os.path.join('data', 'unmatched_in_merged.csv')
            keep = [c for c in ['cname', 'name', 'positions', 'team_name', 'reason'] if c in missing.columns]
            missing[keep].to_csv(out_missing, index=False)
            print(f"Saved unmatched details to {out_missing} ({len(missing)} rows)")
        else:
            # Clean up old report if it exists and we are now 100% matched
            out_missing = os.path.join('data', 'unmatched_in_merged.csv')
            if os.path.exists(out_missing):
                os.remove(out_missing)

        # Summary report
        report = {
            'expected_total_from_yahoo_rosters': len(yinfo),
            'skaters_with_team_name': sk_with_team,
            'goalies_with_team_name': g_with_team,
            'total_with_team_name': total_with_team,
            'difference': len(yinfo) - total_with_team
        }
        
        os.makedirs('data', exist_ok=True)
        with open(os.path.join('data','dq_merge_report.json'), 'w', encoding='utf-8') as f:
            _json.dump(report, f, indent=2)
        print(f"DQ report: {report}")

    except Exception as _e:
        print(f"[Warn] Could not produce DQ report: {_e}")
