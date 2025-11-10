import os

import pandas as pd

from helpers.normalization import normalize_name, normalize_position, to_team_code, to_fpos






def _load_yahoo_all_rosters(csv_path: str) -> pd.DataFrame:
    """
    Load Yahoo all_rosters.csv and return per-player rows with:
      - cname: normalized name for joining
      - ypos: normalized primary position for joining (C/L/R/D/G)
      - fpos: coarse position for fallback joins (F/D/G) where C/L/R → F
      - yahoo_name: the Yahoo-displayed player name
      - yahoo_positions: semicolon-joined eligible positions from Yahoo, with 'Util' removed
      - team_name: Yahoo fantasy team name that currently rosters the player
    Joining elsewhere will be on (cname, ypos), with fallbacks to (cname, fpos) and (cname).
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=['cname', 'ypos', 'fpos', 'yahoo_name', 'yahoo_positions', 'team_name'])
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

        # Aggregate to one row per (cname, ypos). fpos is derived deterministically from ypos.
        agg = (
            ydf.dropna(subset=['cname', 'ypos'])
               .groupby(['cname', 'ypos'], as_index=False)
               .agg(yahoo_name=('yahoo_name', first_non_null),
                    yahoo_positions=('yahoo_positions', agg_positions),
                    team_name=('team_name', first_non_null))
        )
        # Recompute fpos from ypos
        agg['fpos'] = agg['ypos'].apply(to_fpos)
        return agg

    return pd.DataFrame(columns=['cname', 'ypos', 'fpos', 'yahoo_name', 'yahoo_positions', 'team_name'])


def merge_role(nst_csv: str, role: str, out_csv: str) -> str:
    """
    Merge NST stats with Yahoo ownership using local normalization (no registry).
    role: 'skater' or 'goalie'
    Returns output CSV path.
    """

    # Load NST
    if not os.path.exists(nst_csv):
        raise FileNotFoundError(f"NST CSV not found: {nst_csv}")
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

    # Prepare NST keys
    df['cname'] = df['Player'].astype(str).apply(normalize_name)
    if role == 'goalie':
        df['Position'] = 'G'
    # Normalize NST team/position using local helpers
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

    # Start merged from NST; no registry
    merged = df.copy()

    # Load Yahoo roster info and merge on cname+position (no team)
    yinfo = _load_yahoo_all_rosters(os.path.join('data', 'all_rosters.csv'))
    if not yinfo.empty:
        if role == 'skater':
            # Primary join on coarse position (F/D) as suggested; compute NST coarse pos
            merged['fpos'] = merged['pos_code'].apply(to_fpos)
            # Prepare Yahoo coarse F/D view and de-duplicate to one row per (cname, fpos)
            y_fg = yinfo[['cname', 'fpos', 'team_name', 'yahoo_positions', 'yahoo_name']].copy()
            y_fg = y_fg.dropna(subset=['cname', 'fpos'])
            y_fg = y_fg[y_fg['fpos'].isin(['F', 'D'])]
            def _agg_positions(series: pd.Series) -> str:
                acc = []
                seen = set()
                for s in series.dropna().astype(str):
                    for p in s.split(';') if s else []:
                        if p and p not in seen:
                            seen.add(p); acc.append(p)
                return ';'.join(acc)
            if not y_fg.empty:
                y_fg = (
                    y_fg.groupby(['cname', 'fpos'], as_index=False)
                        .agg(team_name=('team_name', 'first'),
                             yahoo_positions=('yahoo_positions', _agg_positions),
                             yahoo_name=('yahoo_name', 'first'))
                )
            # Join on coarse key
            merged = merged.merge(
                y_fg,
                on=['cname', 'fpos'],
                how='left'
            )
            # Skaters: intentionally skip by-name fallback to avoid mixing F vs D for duplicate names
            # Any remaining unmatched rows will keep team_name as null and be listed in unmatched report.
            # Remove merge helpers
            for helper in ['ypos', 'fpos']:
                if helper in merged.columns:
                    merged.drop(columns=[helper], inplace=True)
        else:
            # Original behavior (goalies): exact match on (cname, exact position) with fallback to coarse and by-name
            merged = merged.merge(
                yinfo,
                left_on=['cname', 'pos_code'],
                right_on=['cname', 'ypos'],
                how='left'
            )
            # Coarse F/D/G fallback for rows still missing team_name
            def _to_fpos(p: str) -> str:
                p = str(p).upper()
                if p in {'C', 'L', 'R'}:
                    return 'F'
                if p in {'D', 'G'}:
                    return p
                return ''
            merged['fpos'] = merged['pos_code'].apply(_to_fpos)
            need_fallback = merged['team_name'].isna()
            if need_fallback.any():
                y_fg = yinfo[['cname', 'fpos', 'team_name', 'yahoo_positions', 'yahoo_name']].copy()
                y_fg = y_fg.dropna(subset=['fpos'])
                # De-duplicate by (cname, fpos) to avoid multiplying rows when a player has multiple yahoo primary positions
                if not y_fg.empty:
                    def _agg_positions(series: pd.Series) -> str:
                        acc = []
                        seen = set()
                        for s in series.dropna().astype(str):
                            for p in s.split(';') if s else []:
                                if p and p not in seen:
                                    seen.add(p); acc.append(p)
                        return ';'.join(acc)
                    y_fg = (
                        y_fg.groupby(['cname', 'fpos'], as_index=False)
                            .agg(team_name=('team_name', 'first'),
                                 yahoo_positions=('yahoo_positions', _agg_positions),
                                 yahoo_name=('yahoo_name', 'first'))
                    )
                merged = merged.merge(
                    y_fg,
                    left_on=['cname', 'fpos'],
                    right_on=['cname', 'fpos'],
                    how='left',
                    suffixes=('', '_fg')
                )
                for col in ['team_name', 'yahoo_positions', 'yahoo_name']:
                    alt = f"{col}_fg"
                    if alt in merged.columns:
                        merged[col] = merged[col].fillna(merged[alt])
                drop_alt = [c for c in ['team_name_fg', 'yahoo_positions_fg', 'yahoo_name_fg'] if c in merged.columns]
                if drop_alt:
                    merged.drop(columns=drop_alt, inplace=True)
            # Final fallback: match on cname only (may be ambiguous for duplicate names)
            need_final = merged['team_name'].isna()
            if need_final.any():
                y_by_name = (
                    yinfo.groupby('cname', as_index=False)
                         .agg(team_name=('team_name', 'first'),
                              yahoo_positions=('yahoo_positions', 'first'),
                              yahoo_name=('yahoo_name', 'first'))
                )
                merged = merged.merge(
                    y_by_name,
                    on='cname',
                    how='left',
                    suffixes=('', '_byname')
                )
                for col in ['team_name', 'yahoo_positions', 'yahoo_name']:
                    alt = f"{col}_byname"
                    if alt in merged.columns:
                        merged[col] = merged[col].fillna(merged[alt])
                drop_alt = [c for c in ['team_name_byname', 'yahoo_positions_byname', 'yahoo_name_byname'] if c in merged.columns]
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
    front = [c for c in ['Player', 'Team', 'team_name', 'Elig_Pos'] if c in merged.columns]
    others = [c for c in merged.columns if c not in front]
    merged = merged[front + others]

    # Special-case note: Elias Pettersson disambiguation happens via position in the join above.

    # Ensure output dir
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return out_csv


def run_merge():
    out1 = merge_role(os.path.join('data', 'skaters_scored.csv'), 'skater', os.path.join('data', 'merged_skaters.csv'))
    out2 = merge_role(os.path.join('data', 'goalies.csv'), 'goalie', os.path.join('data', 'merged_goalies.csv'))
    print(f"Saved merged skaters to {out1}")
    print(f"Saved merged goalies to {out2}")

    # If prior-season inputs exist, integrate last-year stats into current files (ly_ prefix)
    try:
        def _coarse_from_elig(elig: str) -> str:
            s = str(elig or '')
            parts = [p.strip() for p in s.split(';') if p.strip()]
            for p in parts:
                c = normalize_position(p)
                f = to_fpos(c)
                if f:
                    return f
            # Fallbacks
            if any(p.upper() == 'D' for p in parts):
                return 'D'
            if any(p.upper() == 'G' for p in parts):
                return 'G'
            return ''

        def _integrate_prior(curr_path: str, role: str, prior_input_csv: str):
            if not os.path.exists(prior_input_csv) or not os.path.exists(curr_path):
                return None
            # Build a prior-season merged view (same shape as current) using existing routine
            tmp_out = os.path.join('data', f'_tmp_prior_merged_{role}s.csv')
            try:
                merge_role(prior_input_csv, role, tmp_out)
                prior_df = pd.read_csv(tmp_out)
            finally:
                try:
                    if os.path.exists(tmp_out):
                        os.remove(tmp_out)
                except Exception:
                    pass
            curr_df = pd.read_csv(curr_path)

            # Build join keys: cname + coarse position (F/D/G) to reduce C/L/R drift
            curr_df['__cname'] = curr_df['Player'].astype(str).apply(normalize_name)
            prior_df['__cname'] = prior_df['Player'].astype(str).apply(normalize_name)
            if role == 'goalie':
                curr_df['__coarse'] = 'G'
                prior_df['__coarse'] = 'G'
            else:
                curr_df['__coarse'] = curr_df.get('Elig_Pos').apply(_coarse_from_elig) if 'Elig_Pos' in curr_df.columns else ''
                prior_df['__coarse'] = prior_df.get('Elig_Pos').apply(_coarse_from_elig) if 'Elig_Pos' in prior_df.columns else ''

            # Prepare prior columns with ly_ prefix (exclude keys to avoid collisions)
            exclude = {'Player', 'Elig_Pos'}
            prior_renamed = {}
            for c in prior_df.columns:
                if c in {'__cname', '__coarse'}:
                    continue
                if c in exclude:
                    prior_renamed[c] = f"ly_{c}"
                else:
                    prior_renamed[c] = f"ly_{c}"
            # Reduce prior to one row per (__cname, __coarse) to avoid cartesian expansion on join
            if not prior_df.empty and {'__cname','__coarse'}.issubset(prior_df.columns):
                sort_cols = [c for c in ['szn_GP_all', 'l7_GP_all'] if c in prior_df.columns]
                if sort_cols:
                    prior_df = prior_df.sort_values(by=sort_cols, ascending=[False]*len(sort_cols))
                # Keep the first (highest GP) per key
                prior_df = prior_df.drop_duplicates(subset=['__cname','__coarse'], keep='first')

            prior_pref = prior_df.rename(columns=prior_renamed)

            # Join
            merged = curr_df.merge(
                prior_pref,
                how='left',
                left_on=['__cname', '__coarse'],
                right_on=['ly___cname', 'ly___coarse'] if 'ly___cname' in prior_pref.columns else ['__cname', '__coarse']
            )
            # Safety: if any duplication still occurred, collapse to unique rows by (Player, Team, Elig_Pos)
            dedup_keys = [k for k in ['Player','Team','Elig_Pos'] if k in merged.columns]
            if dedup_keys:
                merged = merged.drop_duplicates(subset=dedup_keys, keep='first')

            # Team change log
            try:
                ly_team_col = 'ly_Team' if 'ly_Team' in merged.columns else None
                if ly_team_col:
                    changed = merged[(merged[ly_team_col].notna()) & (merged.get('Team') != merged[ly_team_col])].copy()
                    if not changed.empty:
                        changed_out = changed[['Player', ly_team_col, 'Team']].copy()
                        changed_out.rename(columns={ly_team_col: 'Prior_Team', 'Team': 'Current_Team'}, inplace=True)
                        changed_out['Role'] = 'G' if role == 'goalie' else 'Skater'
                        os.makedirs('data', exist_ok=True)
                        tlog = os.path.join('data', 'team_changes.csv')
                        # Append or write
                        if os.path.exists(tlog):
                            prev = pd.read_csv(tlog)
                            combined = pd.concat([prev, changed_out], ignore_index=True).drop_duplicates()
                            combined.to_csv(tlog, index=False)
                        else:
                            changed_out.to_csv(tlog, index=False)
            except Exception as _tc:
                print(f"[Warn] Could not write team change log: {_tc}")

            # Clean helper cols
            for c in ['__cname', '__coarse']:
                if c in merged.columns:
                    del merged[c]
                if f"ly_{c}" in merged.columns:
                    del merged[f"ly_{c}"]

            merged.to_csv(curr_path, index=False)
            return curr_path

        prior_sk_in = os.path.join('data', 'skaters_scored_prior.csv')
        prior_g_in = os.path.join('data', 'goalies_prior.csv')
        _integrate_prior(out1, 'skater', prior_sk_in)
        _integrate_prior(out2, 'goalie', prior_g_in)
    except Exception as _pm:
        print(f"[Warn] Prior-season integration skipped due to error: {_pm}")

    # Lightweight data-quality report for merge coverage
    try:
        import json as _json
        all_rosters_path = os.path.join('data', 'all_rosters.csv')
        if os.path.exists(all_rosters_path):
            ydf = pd.read_csv(all_rosters_path)
            total_yahoo_rows = len(ydf)
            # Build cname+ypos keys for yahoo to better reflect distinct roster spots
            ydf['cname'] = ydf['name'].fillna('').apply(normalize_name)
            def _ypos_map(row):
                sel = str(row.get('selected_position') or '').upper()
                poss = str(row.get('positions') or '')
                parts = [p.strip().upper() for p in poss.split(';') if p.strip()]
                pref = sel if sel and sel not in {'BN','IR','IR+','NA'} else ''
                order = ([pref] if pref else []) + parts
                for p in order:
                    c = normalize_position(p)
                    if c in {'G','D','C','L','R'}:
                        return c
                if 'D' in parts: return 'D'
                if 'G' in parts: return 'G'
                return ''
            ydf['ypos'] = ydf.apply(_ypos_map, axis=1)
            # Distinct players count as in the CSV (user expectation)
            expected_total = total_yahoo_rows

            ms = pd.read_csv(out1)
            mg = pd.read_csv(out2)
            sk_with_team = int(ms['team_name'].notna().sum()) if 'team_name' in ms.columns else 0
            g_with_team = int(mg['team_name'].notna().sum()) if 'team_name' in mg.columns else 0
            total_with_team = sk_with_team + g_with_team

            report = {
                'expected_total_from_all_rosters_rows': expected_total,
                'skaters_with_team_name': sk_with_team,
                'goalies_with_team_name': g_with_team,
                'total_with_team_name': total_with_team,
                'difference': expected_total - total_with_team
            }
            os.makedirs('data', exist_ok=True)
            with open(os.path.join('data','dq_merge_report.json'), 'w', encoding='utf-8') as f:
                _json.dump(report, f, indent=2)
            print(f"DQ report: {report}")

            # Build list of yahoo players not matched in merged files (by cname)
            ms_keys = set(ms['Player'].astype(str).apply(normalize_name)) if 'Player' in ms.columns else set()
            mg_keys = set(mg['Player'].astype(str).apply(normalize_name)) if 'Player' in mg.columns else set()
            merged_keys_with_team = set(ms.loc[ms.get('team_name').notna() if 'team_name' in ms.columns else [], 'Player'].astype(str).apply(normalize_name)) | \
                                     set(mg.loc[mg.get('team_name').notna() if 'team_name' in mg.columns else [], 'Player'].astype(str).apply(normalize_name))
            missing = ydf[~ydf['cname'].isin(merged_keys_with_team)].copy()
            missing_cols = ['name', 'selected_position', 'positions', 'team_name']
            keep_cols = [c for c in missing_cols if c in missing.columns]
            missing_out = missing[['cname'] + keep_cols].drop_duplicates()
            missing_out.to_csv(os.path.join('data','unmatched_in_merged.csv'), index=False)
            print(f"Saved unmatched details to data/unmatched_in_merged.csv ({len(missing_out)} rows)")
        else:
            print("all_rosters.csv not found; skipped DQ merge report.")
    except Exception as _e:
        print(f"[Warn] Could not produce DQ report: {_e}")
