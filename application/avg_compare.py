import math
import os
from typing import Dict, List, Tuple, Optional

from sqlalchemy import func, select

from infrastructure.persistence import (
    get_session,
    League,
    Team,
    StatCategory,
    WeeklyTotal,
    WeeklyPlayerGP,
    PlayerPosition,
)


# Stats where lower is better (invert xWin comparison)
INVERSE_STATS = {"GA", "GAA"}

# Stats to exclude from comparison grids (informational only)
EXCLUDE_STATS = {"SA", "SV"}


# Default grouping and ordering (used if DB lacks grouping metadata)
SKATER_GROUP_ORDER = [
    ("Offensive", ["G", "A", "PPP", "SOG", "FW"]),
    ("Banger", ["HIT", "BLK", "PIM"]),
    ("Other", ["+/-", "SHP"]),
]
GOALIE_ORDER = ["W", "GA", "GAA", "SV%", "SHO"]


def _find_team_key(session, league_id: int, team_id_or_key: str) -> str:
    """Resolve a user-friendly team identifier into a canonical team_key.

    Accepts either a full Yahoo-like team_key (e.g., nhl.p.2526.t.4) or a
    numeric team_id (e.g., "4").
    """
    # If it already matches an existing team_key in this league, return it
    t = (
        session.query(Team)
        .filter(Team.league_id == league_id, Team.team_key == str(team_id_or_key))
        .one_or_none()
    )
    if t:
        return t.team_key

    # Fallback: match by suffix ".t.<id>"
    suffix = f".t.{int(team_id_or_key)}"
    t = (
        session.query(Team)
        .filter(Team.league_id == league_id, Team.team_key.like(f"%{suffix}"))
        .one_or_none()
    )
    if not t:
        raise SystemExit(f"Team not found for identifier: {team_id_or_key}")
    return t.team_key


def _avg_stats_by_team(session, league_id: int, team_key: str, current_week: int) -> Dict[int, float]:
    """Average of WeeklyTotal.value by stat_id across weeks < current_week."""
    q = (
        session.query(WeeklyTotal.stat_id, func.avg(WeeklyTotal.value))
        .filter(
            WeeklyTotal.league_id == league_id,
            WeeklyTotal.team_key == team_key,
            WeeklyTotal.week_num < int(current_week),
        )
        .group_by(WeeklyTotal.stat_id)
    )
    return {sid: float(val) if val is not None else math.nan for sid, val in q.all()}


def _avg_stats_league(session, league_id: int, current_week: int) -> Dict[int, float]:
    q = (
        session.query(WeeklyTotal.stat_id, func.avg(WeeklyTotal.value))
        .filter(
            WeeklyTotal.league_id == league_id,
            WeeklyTotal.week_num < int(current_week),
        )
        .group_by(WeeklyTotal.stat_id)
    )
    return {sid: float(val) if val is not None else math.nan for sid, val in q.all()}


def _avg_gp_by_pos_type(session, league_id: int, team_key: str, current_week: int) -> Dict[str, float]:
    """Average GP per week by position_type using WeeklyPlayerGP joined to PlayerPosition.

    position_type keys: 'P' (skaters) and 'G' (goalies).
    """
    # Sum GP by week for each player and classify goalie if has position 'G'
    # Use EXISTS subquery for goalie classification
    pp_alias = select(PlayerPosition.player_key).where(PlayerPosition.position == 'G').subquery()
    # Weekly sums for goalies
    goalie_q = (
        session.query(WeeklyPlayerGP.week_num, func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(
            WeeklyPlayerGP.league_id == league_id,
            WeeklyPlayerGP.team_key == team_key,
            WeeklyPlayerGP.week_num < int(current_week),
            WeeklyPlayerGP.player_key.in_(select(pp_alias.c.player_key)),
        )
        .group_by(WeeklyPlayerGP.week_num)
    )
    goalie_by_week = {wk: int(val or 0) for wk, val in goalie_q.all()}

    # Weekly sums for skaters = all - goalies
    all_q = (
        session.query(WeeklyPlayerGP.week_num, func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(
            WeeklyPlayerGP.league_id == league_id,
            WeeklyPlayerGP.team_key == team_key,
            WeeklyPlayerGP.week_num < int(current_week),
        )
        .group_by(WeeklyPlayerGP.week_num)
    )
    all_by_week = {wk: int(val or 0) for wk, val in all_q.all()}

    weeks = sorted(all_by_week.keys())
    if not weeks:
        return {'P': 0.0, 'G': 0.0}
    g_avgs = sum(goalie_by_week.get(wk, 0) for wk in weeks) / len(weeks)
    p_avgs = sum(max(all_by_week.get(wk, 0) - goalie_by_week.get(wk, 0), 0) for wk in weeks) / len(weeks)
    return {'P': float(g_avgs if False else p_avgs), 'G': float(g_avgs)}  # placeholder replaced below


def _avg_gp_by_pos_type(session, league_id: int, team_key: str, current_week: int) -> Dict[str, float]:
    # Re-implement without placeholder bug: compute both and return
    pp_goalie = select(PlayerPosition.player_key).where(PlayerPosition.position == 'G').subquery()

    all_q = (
        session.query(WeeklyPlayerGP.week_num, func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(
            WeeklyPlayerGP.league_id == league_id,
            WeeklyPlayerGP.team_key == team_key,
            WeeklyPlayerGP.week_num < int(current_week),
        )
        .group_by(WeeklyPlayerGP.week_num)
    )
    all_by_week = {wk: int(val or 0) for wk, val in all_q.all()}

    g_q = (
        session.query(WeeklyPlayerGP.week_num, func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(
            WeeklyPlayerGP.league_id == league_id,
            WeeklyPlayerGP.team_key == team_key,
            WeeklyPlayerGP.week_num < int(current_week),
            WeeklyPlayerGP.player_key.in_(select(pp_goalie.c.player_key)),
        )
        .group_by(WeeklyPlayerGP.week_num)
    )
    g_by_week = {wk: int(val or 0) for wk, val in g_q.all()}

    weeks = sorted(all_by_week.keys())
    if not weeks:
        return {'P': 0.0, 'G': 0.0}
    g_avg = sum(g_by_week.get(wk, 0) for wk in weeks) / len(weeks)
    p_avg = sum(max(all_by_week.get(wk, 0) - g_by_week.get(wk, 0), 0) for wk in weeks) / len(weeks)
    return {'P': float(p_avg), 'G': float(g_avg)}


def _league_avg_gp_by_pos_type(session, league_id: int, current_week: int) -> Dict[str, float]:
    # Average across teams: compute team-week sums first then mean by week and then average across weeks
    pp_goalie = select(PlayerPosition.player_key).where(PlayerPosition.position == 'G').subquery()

    # All team-week totals
    all_tw = (
        session.query(WeeklyPlayerGP.team_key, WeeklyPlayerGP.week_num, func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(WeeklyPlayerGP.league_id == league_id, WeeklyPlayerGP.week_num < int(current_week))
        .group_by(WeeklyPlayerGP.team_key, WeeklyPlayerGP.week_num)
        .all()
    )
    g_tw = (
        session.query(WeeklyPlayerGP.team_key, WeeklyPlayerGP.week_num, func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(
            WeeklyPlayerGP.league_id == league_id,
            WeeklyPlayerGP.week_num < int(current_week),
            WeeklyPlayerGP.player_key.in_(select(pp_goalie.c.player_key)),
        )
        .group_by(WeeklyPlayerGP.team_key, WeeklyPlayerGP.week_num)
        .all()
    )
    # Index by (team, week)
    all_map: Dict[Tuple[str, int], int] = {(t, w): int(v or 0) for t, w, v in all_tw}
    g_map: Dict[Tuple[str, int], int] = {(t, w): int(v or 0) for t, w, v in g_tw}

    # Weeks present
    weeks = sorted({w for _, w in all_map.keys()})
    if not weeks:
        return {'P': 0.0, 'G': 0.0}

    # For each week, compute average per team of G and P
    g_week_means: List[float] = []
    p_week_means: List[float] = []
    for wk in weeks:
        teams = {t for (t, w) in all_map.keys() if w == wk}
        if not teams:
            continue
        g_vals = [g_map.get((t, wk), 0) for t in teams]
        p_vals = [max(all_map.get((t, wk), 0) - g_map.get((t, wk), 0), 0) for t in teams]
        g_week_means.append(sum(g_vals) / len(g_vals))
        p_week_means.append(sum(p_vals) / len(p_vals))

    return {
        'P': float(sum(p_week_means) / len(p_week_means)) if p_week_means else 0.0,
        'G': float(sum(g_week_means) / len(g_week_means)) if g_week_means else 0.0,
    }


def _sum_stats_by_team(session, league_id: int, team_key: str, current_week: int) -> Dict[int, float]:
    """Sum of WeeklyTotal.value by stat_id across weeks < current_week (for per-GP grid)."""
    q = (
        session.query(WeeklyTotal.stat_id, func.sum(WeeklyTotal.value))
        .filter(
            WeeklyTotal.league_id == league_id,
            WeeklyTotal.team_key == team_key,
            WeeklyTotal.week_num < int(current_week),
        )
        .group_by(WeeklyTotal.stat_id)
    )
    return {sid: float(val or 0.0) for sid, val in q.all()}


def _sum_stats_league(session, league_id: int, current_week: int) -> Dict[int, float]:
    q = (
        session.query(WeeklyTotal.stat_id, func.sum(WeeklyTotal.value))
        .filter(
            WeeklyTotal.league_id == league_id,
            WeeklyTotal.week_num < int(current_week),
        )
        .group_by(WeeklyTotal.stat_id)
    )
    return {sid: float(val or 0.0) for sid, val in q.all()}


def _total_gp_by_pos_type(session, league_id: int, team_key: str, current_week: int) -> Dict[str, float]:
    """Total GP prior to current week by position_type ('P','G')."""
    pp_goalie = select(PlayerPosition.player_key).where(PlayerPosition.position == 'G').subquery()

    all_q = (
        session.query(func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(
            WeeklyPlayerGP.league_id == league_id,
            WeeklyPlayerGP.team_key == team_key,
            WeeklyPlayerGP.week_num < int(current_week),
        )
    )
    all_total = int(all_q.scalar() or 0)

    g_q = (
        session.query(func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(
            WeeklyPlayerGP.league_id == league_id,
            WeeklyPlayerGP.team_key == team_key,
            WeeklyPlayerGP.week_num < int(current_week),
            WeeklyPlayerGP.player_key.in_(select(pp_goalie.c.player_key)),
        )
    )
    g_total = int(g_q.scalar() or 0)
    p_total = max(all_total - g_total, 0)
    return {'P': float(p_total), 'G': float(g_total)}


def _league_total_gp_by_pos_type(session, league_id: int, current_week: int) -> Dict[str, float]:
    pp_goalie = select(PlayerPosition.player_key).where(PlayerPosition.position == 'G').subquery()

    all_total = (
        session.query(func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(WeeklyPlayerGP.league_id == league_id, WeeklyPlayerGP.week_num < int(current_week))
        .scalar()
    )
    g_total = (
        session.query(func.coalesce(func.sum(WeeklyPlayerGP.gp), 0))
        .filter(
            WeeklyPlayerGP.league_id == league_id,
            WeeklyPlayerGP.week_num < int(current_week),
            WeeklyPlayerGP.player_key.in_(select(pp_goalie.c.player_key)),
        )
        .scalar()
    )
    all_total = int(all_total or 0)
    g_total = int(g_total or 0)
    p_total = max(all_total - g_total, 0)
    return {'P': float(p_total), 'G': float(g_total)}


def _downloads_default_path(filename: str) -> str:
    home = os.path.expanduser("~")
    dlds = os.path.join(home, "Downloads")
    try:
        os.makedirs(dlds, exist_ok=True)
    except Exception:
        pass
    return os.path.join(dlds, filename)


def _stat_sort_key_factory(
    pos: str,
    abbr_to_meta: Dict[str, Tuple[Optional[str], Optional[int], Optional[int]]],
):
    """Return a key function to sort StatCategory by group and stat order.

    abbr_to_meta maps abbr -> (group_code, group_order, stat_order)
    Fallback to predefined mappings if metadata missing.
    """
    # Build fallback map: abbr -> (grp_idx, stat_idx)
    fb_map: Dict[str, Tuple[int, int]] = {}
    if pos == 'P':
        for gi, (gname, stats) in enumerate(SKATER_GROUP_ORDER):
            for si, ab in enumerate(stats):
                fb_map[ab.upper()] = (gi, si)
    else:
        for si, ab in enumerate(GOALIE_ORDER):
            fb_map[ab.upper()] = (0, si)

    def key_fn(stat: StatCategory):
        ab = (stat.abbr or "").upper()
        meta = abbr_to_meta.get(ab)
        if meta and meta[1] is not None and meta[2] is not None:
            return (int(meta[1]), int(meta[2]), ab)
        # fallback
        if ab in fb_map:
            gi, si = fb_map[ab]
            return (gi, si, ab)
        # generic: place after known groups, order by abbr then id
        return (99, 99, ab or str(stat.stat_id))

    return key_fn


def compare_averages(*, league_key: str, current_week: int, team_id: str, opp_team_id: str):
    """Compute and print average stat values prior to current week for a team.

    Produces two side-by-side comparisons in a single grid section per position_type (P, G):
    - Team vs Opponent
    - Team vs League Average
    Also includes average GP rows derived from WeeklyPlayerGP.
    """
    session = get_session()
    try:
        league = session.query(League).filter(League.league_key == str(league_key)).one_or_none()
        if league is None:
            raise SystemExit(f"League not found: {league_key}")

        team_key = _find_team_key(session, league.id, str(team_id))
        opp_key = _find_team_key(session, league.id, str(opp_team_id))

        # Stat categories by pos_type (pull grouping metadata if present)
        stats = (
            session.query(StatCategory)
            .filter(StatCategory.league_id == league.id)
            .all()
        )
        by_pos: Dict[str, List[StatCategory]] = {'P': [], 'G': []}
        abbr_meta: Dict[str, Tuple[Optional[str], Optional[int], Optional[int]]] = {}
        for s in stats:
            if s.position_type in ('P', 'G'):
                by_pos[s.position_type].append(s)
            abbr = (s.abbr or "").upper()
            # Not all DBs will have these columns yet; getattr with default None
            group_code = getattr(s, 'group_code', None)
            group_order = getattr(s, 'group_order', None)
            stat_order = getattr(s, 'stat_order', None)
            abbr_meta[abbr] = (group_code, group_order, stat_order)

        # Quick lookup for special goalie computations
        abbr_to_id: Dict[str, int] = { (s.abbr or '').upper(): s.stat_id for s in stats }

        # Pre-compute averages
        team_avg = _avg_stats_by_team(session, league.id, team_key, current_week)
        opp_avg = _avg_stats_by_team(session, league.id, opp_key, current_week)
        league_avg = _avg_stats_league(session, league.id, current_week)

        # GP averages (informational footer)
        team_gp = _avg_gp_by_pos_type(session, league.id, team_key, current_week)
        opp_gp = _avg_gp_by_pos_type(session, league.id, opp_key, current_week)
        league_gp = _league_avg_gp_by_pos_type(session, league.id, current_week)

        # Points calculation (if league has point values)
        has_points = any(s.value is not None and float(s.value) != 0 for s in stats)
        stat_weights: Dict[int, float] = {s.stat_id: float(s.value) for s in stats if s.value is not None}

        # Totals for per-GP grid
        team_sum = _sum_stats_by_team(session, league.id, team_key, current_week)
        opp_sum = _sum_stats_by_team(session, league.id, opp_key, current_week)
        league_sum = _sum_stats_league(session, league.id, current_week)
        team_gp_tot = _total_gp_by_pos_type(session, league.id, team_key, current_week)
        opp_gp_tot = _total_gp_by_pos_type(session, league.id, opp_key, current_week)
        league_gp_tot = _league_total_gp_by_pos_type(session, league.id, current_week)

        # Build CSV rows
        csv_rows: List[Dict[str, object]] = []

        for pos in ('P', 'G'):
            # Sorting using grouping metadata with fallback
            sort_key = _stat_sort_key_factory(pos, abbr_meta)

            base_rows: List[Tuple[str, float, float, float, int, float, float, float, int, str, float, float, float]] = []
            for s in sorted(by_pos.get(pos, []), key=sort_key):
                name = s.abbr or s.name or str(s.stat_id)
                if (s.abbr or "").upper() in EXCLUDE_STATS:
                    continue
                a = float(team_avg.get(s.stat_id, float('nan')))
                b = float(opp_avg.get(s.stat_id, float('nan')))
                lavg = float(league_avg.get(s.stat_id, float('nan')))
                d1 = a - b
                d2 = a - lavg
                inv = (s.abbr or "").upper() in INVERSE_STATS
                win_vs_opp = 1 if (a < b if inv else a > b) else 0
                win_vs_lg = 1 if (a < lavg if inv else a > lavg) else 0
                group_code = abbr_meta.get((s.abbr or '').upper(), (None, None, None))[0]
                
                # Points contribution per game
                w = stat_weights.get(s.stat_id, 0.0)
                a_pts = a * w if not math.isnan(a) else 0.0
                b_pts = b * w if not math.isnan(b) else 0.0
                l_pts = lavg * w if not math.isnan(lavg) else 0.0
                
                base_rows.append((name, a, b, d1, win_vs_opp, a, lavg, d2, win_vs_lg, group_code or ("Goalie" if pos == 'G' else ""), a_pts, b_pts, l_pts))

            # Per-GP grid
            rows_gp: List[Tuple[str, float, float, float, int, float, float, float, int, str, float, float, float]] = []
            t_gp = float(team_gp_tot.get(pos, 0.0))
            o_gp = float(opp_gp_tot.get(pos, 0.0))
            l_gp = float(league_gp_tot.get(pos, 0.0))
            for s in sorted(by_pos.get(pos, []), key=sort_key):
                name = s.abbr or s.name or str(s.stat_id)
                if (s.abbr or "").upper() in EXCLUDE_STATS:
                    continue
                a_tot = float(team_sum.get(s.stat_id, 0.0))
                b_tot = float(opp_sum.get(s.stat_id, 0.0))
                l_tot = float(league_sum.get(s.stat_id, 0.0))
                ab = (s.abbr or '').upper()
                if ab == 'GAA':
                    # Recompute using GA totals divided by goalie GP
                    ga_id = abbr_to_id.get('GA')
                    a_num = float(team_sum.get(ga_id, 0.0)) if ga_id is not None else 0.0
                    b_num = float(opp_sum.get(ga_id, 0.0)) if ga_id is not None else 0.0
                    l_num = float(league_sum.get(ga_id, 0.0)) if ga_id is not None else 0.0
                    a = (a_num / t_gp) if t_gp > 0 else float('nan')
                    b = (b_num / o_gp) if o_gp > 0 else float('nan')
                    lavg = (l_num / l_gp) if l_gp > 0 else float('nan')
                elif ab == 'SV%':
                    # Recompute using SV / SA
                    sv_id = abbr_to_id.get('SV')
                    sa_id = abbr_to_id.get('SA')
                    def safe_ratio(n: float, d: float) -> float:
                        return (n / d) if d > 0 else float('nan')
                    a = safe_ratio(float(team_sum.get(sv_id, 0.0)) if sv_id is not None else 0.0,
                                   float(team_sum.get(sa_id, 0.0)) if sa_id is not None else 0.0)
                    b = safe_ratio(float(opp_sum.get(sv_id, 0.0)) if sv_id is not None else 0.0,
                                   float(opp_sum.get(sa_id, 0.0)) if sa_id is not None else 0.0)
                    lavg = safe_ratio(float(league_sum.get(sv_id, 0.0)) if sv_id is not None else 0.0,
                                      float(league_sum.get(sa_id, 0.0)) if sa_id is not None else 0.0)
                else:
                    a = (a_tot / t_gp) if t_gp > 0 else float('nan')
                    b = (b_tot / o_gp) if o_gp > 0 else float('nan')
                    lavg = (l_tot / l_gp) if l_gp > 0 else float('nan')
                d1 = a - b
                d2 = a - lavg
                inv = (s.abbr or "").upper() in INVERSE_STATS
                win_vs_opp = 1 if (a < b if inv else a > b) else 0
                win_vs_lg = 1 if (a < lavg if inv else a > lavg) else 0
                group_code = abbr_meta.get((s.abbr or '').upper(), (None, None, None))[0]
                
                # Points contribution per game (avg stats * weight)
                w = stat_weights.get(s.stat_id, 0.0)
                a_pts = a * w if not math.isnan(a) else 0.0
                b_pts = b * w if not math.isnan(b) else 0.0
                l_pts = lavg * w if not math.isnan(lavg) else 0.0
                
                rows_gp.append((name, a, b, d1, win_vs_opp, a, lavg, d2, win_vs_lg, group_code or ("Goalie" if pos == 'G' else ""), a_pts, b_pts, l_pts))

            # Emit CSV rows for this section
            section = 'Skaters' if pos == 'P' else 'Goalies'
            # Base grid rows
            total_wins_vs_opp = sum(r[4] for r in base_rows)
            total_wins_vs_lg = sum(r[8] for r in base_rows)
            nstats = len(base_rows) or 1
            total_pts_team = sum(r[10] for r in base_rows)
            total_pts_opp = sum(r[11] for r in base_rows)
            total_pts_lg = sum(r[12] for r in base_rows)

            for (name, a, b, d1, w1, a2, lavg, d2, w2, grp, a_pts, b_pts, l_pts) in base_rows:
                row = {
                    'section': section,
                    'grid': 'base',
                    'group': grp,
                    'stat': name,
                    'team': a,
                    'opp': b,
                    'diff_vs_opp': d1,
                    'xWin_vs_opp': w1,
                    'team_vs_league': a2,
                    'league_avg': lavg,
                    'diff_vs_league': d2,
                    'xWin_vs_league': w2,
                }
                if has_points:
                    row['team_pts'] = a_pts
                    row['opp_pts'] = b_pts
                    row['league_pts'] = l_pts
                csv_rows.append(row)
            
            # Totals and Win%
            total_row = {'section': section, 'grid': 'base', 'group': '', 'stat': 'Total Wins', 'team': total_wins_vs_opp, 'opp': '', 'diff_vs_opp': '', 'xWin_vs_opp': '', 'team_vs_league': total_wins_vs_lg, 'league_avg': '', 'diff_vs_league': '', 'xWin_vs_league': ''}
            if has_points:
                total_row['team_pts'] = total_pts_team
                total_row['opp_pts'] = total_pts_opp
                total_row['league_pts'] = total_pts_lg
            csv_rows.append(total_row)
            
            win_pct_row = {'section': section, 'grid': 'base', 'group': '', 'stat': 'Win%', 'team': (total_wins_vs_opp / nstats), 'opp': '', 'diff_vs_opp': '', 'xWin_vs_opp': '', 'team_vs_league': (total_wins_vs_lg / nstats), 'league_avg': '', 'diff_vs_league': '', 'xWin_vs_league': ''}
            csv_rows.append(win_pct_row)

            # Avg GP footer
            avg_gp_row = {'section': section, 'grid': 'base', 'group': '', 'stat': 'Avg GP', 'team': float(team_gp.get(pos, 0.0)), 'opp': float(opp_gp.get(pos, 0.0)), 'diff_vs_opp': '', 'xWin_vs_opp': '', 'team_vs_league': float(team_gp.get(pos, 0.0)), 'league_avg': float(league_gp.get(pos, 0.0)), 'diff_vs_league': '', 'xWin_vs_league': ''}
            csv_rows.append(avg_gp_row)

            # Per-GP grid rows
            total_wins_vs_opp_gp = sum(r[4] for r in rows_gp)
            total_wins_vs_lg_gp = sum(r[8] for r in rows_gp)
            nstats_gp = len(rows_gp) or 1
            total_pts_team_gp = sum(r[10] for r in rows_gp)
            total_pts_opp_gp = sum(r[11] for r in rows_gp)
            total_pts_lg_gp = sum(r[12] for r in rows_gp)

            for (name, a, b, d1, w1, a2, lavg, d2, w2, grp, a_pts, b_pts, l_pts) in rows_gp:
                row = {
                    'section': section,
                    'grid': 'per_gp',
                    'group': grp,
                    'stat': name,
                    'team': a,
                    'opp': b,
                    'diff_vs_opp': d1,
                    'xWin_vs_opp': w1,
                    'team_vs_league': a2,
                    'league_avg': lavg,
                    'diff_vs_league': d2,
                    'xWin_vs_league': w2,
                }
                if has_points:
                    row['team_pts'] = a_pts
                    row['opp_pts'] = b_pts
                    row['league_pts'] = l_pts
                csv_rows.append(row)
            
            total_row_gp = {'section': section, 'grid': 'per_gp', 'group': '', 'stat': 'Total Wins', 'team': total_wins_vs_opp_gp, 'opp': '', 'diff_vs_opp': '', 'xWin_vs_opp': '', 'team_vs_league': total_wins_vs_lg_gp, 'league_avg': '', 'diff_vs_league': '', 'xWin_vs_league': ''}
            if has_points:
                total_row_gp['team_pts'] = total_pts_team_gp
                total_row_gp['opp_pts'] = total_pts_opp_gp
                total_row_gp['league_pts'] = total_pts_lg_gp
            csv_rows.append(total_row_gp)

            win_pct_row_gp = {'section': section, 'grid': 'per_gp', 'group': '', 'stat': 'Win%', 'team': (total_wins_vs_opp_gp / nstats_gp), 'opp': '', 'diff_vs_opp': '', 'xWin_vs_opp': '', 'team_vs_league': (total_wins_vs_lg_gp / nstats_gp), 'league_avg': '', 'diff_vs_league': '', 'xWin_vs_league': ''}
            csv_rows.append(win_pct_row_gp)

        # Write CSV
        safe_lkey = str(league_key).replace(':', '_').replace('/', '_')
        filename = f"avg_compare_{safe_lkey}_w{current_week}_t{team_id}_vs_{opp_team_id}.csv"
        out_path = _downloads_default_path(filename)
        # Write manually to avoid pandas dependency (already present but keep light)
        import csv
        fieldnames = ['section', 'grid', 'group', 'stat', 'team', 'opp', 'diff_vs_opp', 'xWin_vs_opp', 'team_vs_league', 'league_avg', 'diff_vs_league', 'xWin_vs_league']
        if has_points:
            fieldnames += ['team_pts', 'opp_pts', 'league_pts']

        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                # filter to only known fields
                filtered = {k: v for k, v in row.items() if k in fieldnames}
                writer.writerow(filtered)

        print(f"avg-compare: Computed Team {team_id} vs Opp {opp_team_id} for weeks < {current_week} in league {league_key}.")
        print("- Excluded stats: SA, SV from grids (info only)")
        print("- Inverse xWin for: GA, GAA (lower is better)")
        print("- Per-GP special handling: GAA = total GA / goalie GP; SV% = total SV / total SA")
        print(f"CSV written to: {out_path}")
    finally:
        session.close()
