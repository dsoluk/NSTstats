import os
import datetime as _dt
from typing import Optional, Iterable, Any

from sqlalchemy import (
    create_engine, Integer, String, Date, Boolean, Enum, ForeignKey, UniqueConstraint,
    Numeric, text
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker, Session


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_key: Mapped[str] = mapped_column(String(32), nullable=False)
    league_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    season: Mapped[str] = mapped_column(String(16), nullable=False)
    last_roster_sync: Mapped[Optional[_dt.datetime]] = mapped_column(nullable=True)
    last_nst_sync: Mapped[Optional[_dt.datetime]] = mapped_column(nullable=True)
    last_sched_sync: Mapped[Optional[_dt.datetime]] = mapped_column(nullable=True)

    weeks: Mapped[list["Week"]] = relationship(back_populates="league")
    teams: Mapped[list["Team"]] = relationship(back_populates="league")
    stats: Mapped[list["StatCategory"]] = relationship(back_populates="league")
    roster_requirements: Mapped[list["RosterRequirement"]] = relationship(back_populates="league")


class Week(Base):
    __tablename__ = "weeks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    week_num: Mapped[int] = mapped_column(Integer, nullable=False)
    start_date: Mapped[Optional[_dt.date]] = mapped_column(Date, nullable=True)
    end_date: Mapped[Optional[_dt.date]] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="open", nullable=False)

    league: Mapped[League] = relationship(back_populates="weeks")

    __table_args__ = (
        UniqueConstraint("league_id", "week_num", name="uq_weeks_league_week"),
    )


class Team(Base):
    __tablename__ = "teams"
    team_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    team_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)

    league: Mapped[League] = relationship(back_populates="teams")


class Player(Base):
    __tablename__ = "players"
    player_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    positions: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    last_synced: Mapped[Optional[_dt.datetime]] = mapped_column(nullable=True)

class CurrentRoster(Base):
    """Stores the current ownership state for the merge and avg-compare processes."""
    __tablename__ = "current_rosters"
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), primary_key=True)
    player_key: Mapped[str] = mapped_column(ForeignKey("players.player_key", ondelete="CASCADE"), primary_key=True)
    team_key: Mapped[str] = mapped_column(ForeignKey("teams.team_key", ondelete="CASCADE"), nullable=False)

class PlayerPosition(Base):
    __tablename__ = "player_positions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_key: Mapped[str] = mapped_column(ForeignKey("players.player_key", ondelete="CASCADE"), nullable=False)
    position: Mapped[str] = mapped_column(String(16), nullable=False)
    __table_args__ = (
        UniqueConstraint("player_key", "position", name="uq_player_position"),
    )


class StatCategory(Base):
    __tablename__ = "stat_categories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    stat_id: Mapped[int] = mapped_column(Integer, nullable=False)
    abbr: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    position_type: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    value: Mapped[Optional[float]] = mapped_column(Numeric(18, 6), nullable=True)
    # Optional grouping metadata for ordering reports
    group_code: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    group_order: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    stat_order: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    league: Mapped[League] = relationship(back_populates="stats")
    __table_args__ = (
        UniqueConstraint("league_id", "stat_id", name="uq_stat_cat_league_stat"),
    )


class RosterRequirement(Base):
    __tablename__ = "roster_requirements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    position: Mapped[str] = mapped_column(String(16), nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    league: Mapped[League] = relationship(back_populates="roster_requirements")
    __table_args__ = (
        UniqueConstraint("league_id", "position", name="uq_roster_req_league_pos"),
    )


class Matchup(Base):
    __tablename__ = "matchups"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    week_num: Mapped[int] = mapped_column(Integer, nullable=False)
    matchup_index: Mapped[int] = mapped_column(Integer, nullable=False)
    __table_args__ = (
        UniqueConstraint("league_id", "week_num", "matchup_index", name="uq_matchup_l_w_m"),
    )


class WeeklyTotal(Base):
    __tablename__ = "weekly_totals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    week_num: Mapped[int] = mapped_column(Integer, nullable=False)
    matchup_id: Mapped[Optional[int]] = mapped_column(ForeignKey("matchups.id", ondelete="SET NULL"), nullable=True)
    team_key: Mapped[str] = mapped_column(ForeignKey("teams.team_key", ondelete="CASCADE"), nullable=False)
    stat_id: Mapped[int] = mapped_column(Integer, nullable=False)
    stat_abbr: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    value: Mapped[Optional[float]] = mapped_column(Numeric(18, 6), nullable=True)
    __table_args__ = (
        UniqueConstraint("league_id", "week_num", "team_key", "stat_id", name="uq_weekly_total_key"),
    )


class WeeklyPlayerGP(Base):
    __tablename__ = "weekly_player_gp"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    week_num: Mapped[int] = mapped_column(Integer, nullable=False)
    team_key: Mapped[str] = mapped_column(ForeignKey("teams.team_key", ondelete="CASCADE"), nullable=False)
    player_key: Mapped[str] = mapped_column(ForeignKey("players.player_key", ondelete="CASCADE"), nullable=False)
    gp: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    computed_at: Mapped[Optional[_dt.datetime]] = mapped_column(nullable=True)
    __table_args__ = (
        UniqueConstraint("league_id", "week_num", "team_key", "player_key", name="uq_weekly_player_gp_key"),
    )


class RosterSlotDaily(Base):
    # test comment
    __tablename__ = "roster_slots_daily"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[_dt.date] = mapped_column(Date, nullable=False)
    league_id: Mapped[int] = mapped_column(ForeignKey("leagues.id", ondelete="CASCADE"), nullable=False)
    team_key: Mapped[str] = mapped_column(ForeignKey("teams.team_key", ondelete="CASCADE"), nullable=False)
    player_key: Mapped[str] = mapped_column(ForeignKey("players.player_key", ondelete="CASCADE"), nullable=False)
    selected_position: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    had_game: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    gp: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    __table_args__ = (
        UniqueConstraint("date", "league_id", "team_key", "player_key", name="uq_roster_slot_daily_key"),
    )


_SessionLocal: Optional[sessionmaker] = None


def _ensure_data_dir_sqlite(url: str):
    if url.startswith("sqlite:///"):
        # Path is relative to CWD
        path = url.replace("sqlite:///", "")
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def get_engine_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///data/app.db")


def get_engine(echo: bool = False):
    url = get_engine_url()
    _ensure_data_dir_sqlite(url)
    # Heroku may provide postgres://; SQLAlchemy prefers postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(url, echo=echo, future=True)
    return engine


def init_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine(echo=False)
        # Try to run Alembic migrations automatically
        try:
            import alembic.config
            import alembic.command
            import os
            
            # Find alembic.ini in project root
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ini_path = os.path.join(root_dir, "alembic.ini")
            
            if os.path.exists(ini_path):
                cfg = alembic.config.Config(ini_path)
                # Ensure the engine URL matches
                cfg.set_main_option("sqlalchemy.url", get_engine_url())
                alembic.command.upgrade(cfg, "head")
        except Exception as e:
            # Fallback: create tables if Alembic isn't available or fails
            print(f"[Warn] Auto-migration failed: {e}. Falling back to metadata.create_all()")
            Base.metadata.create_all(engine)
            
        _SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    return _SessionLocal


def get_session() -> Session:
    return init_session_factory()()


# Upsert helpers
def upsert_league(session: Session, *, game_key: str, league_key: str, season: str) -> League:
    obj = session.query(League).filter_by(league_key=league_key).one_or_none()
    if obj is None:
        obj = League(game_key=game_key, league_key=league_key, season=str(season))
        session.add(obj)
        session.flush()
    else:
        changed = False
        if obj.game_key != game_key:
            obj.game_key = game_key; changed = True
        if obj.season != str(season):
            obj.season = str(season); changed = True
        if changed:
            session.flush()
    return obj


def upsert_week(session: Session, *, league_id: int, week_num: int,
                start_date: Optional[_dt.date], end_date: Optional[_dt.date]) -> Week:
    obj = session.query(Week).filter_by(league_id=league_id, week_num=week_num).one_or_none()
    if obj is None:
        obj = Week(league_id=league_id, week_num=week_num, start_date=start_date, end_date=end_date, status="open")
        session.add(obj)
        session.flush()
    else:
        changed = False
        if (start_date and obj.start_date != start_date):
            obj.start_date = start_date; changed = True
        if (end_date and obj.end_date != end_date):
            obj.end_date = end_date; changed = True
        if changed:
            session.flush()
    return obj


def mark_week_closed(session: Session, *, league_id: int, week_num: int) -> None:
    obj = session.query(Week).filter_by(league_id=league_id, week_num=week_num).one_or_none()
    if obj is None:
        obj = Week(league_id=league_id, week_num=week_num, status="closed")
        session.add(obj)
    else:
        obj.status = "closed"
    session.flush()


def is_week_closed(session: Session, *, league_id: int, week_num: int) -> bool:
    obj = session.query(Week).filter_by(league_id=league_id, week_num=week_num).one_or_none()
    return (obj is not None) and (obj.status == "closed")


def upsert_team(session, league_id: int, team_key: str, team_name: str):
    # upsert statement on conflict do update set
    stmt = text("""
        INSERT INTO teams (league_id, team_key, team_name)
        VALUES (:league_id, :team_key, :team_name)
        ON CONFLICT (team_key) DO UPDATE SET
            league_id = :league_id,
            team_name = :team_name
    """)
    session.execute(stmt, {"league_id": league_id, "team_key": team_key, "team_name": team_name})
    return session.query(Team).filter(Team.team_key == team_key).one()

def upsert_player(session, player_key: str, name: str, positions: str, status: Optional[str] = None):
    """Update or create a player record with current metadata."""
    player = session.query(Player).filter(Player.player_key == player_key).one_or_none()
    if not player:
        player = Player(player_key=player_key)
        session.add(player)
    player.name = name
    player.positions = positions
    player.status = status
    player.last_synced = _dt.datetime.now()
    return player

def set_player_positions(session: Session, *, player_key: str, positions: Iterable[str]) -> None:
    # Upsert unique set
    pos_set = {p for p in (positions or []) if str(p).strip()}
    # Delete missing
    session.query(PlayerPosition).filter(PlayerPosition.player_key == player_key, ~PlayerPosition.position.in_(pos_set)).delete(synchronize_session=False)
    # Insert new
    for p in pos_set:
        exists = session.query(PlayerPosition).filter_by(player_key=player_key, position=p).one_or_none()
        if exists is None:
            session.add(PlayerPosition(player_key=player_key, position=p))
    session.flush()


def upsert_stat_category(session: Session, *, league_id: int, stat_id: int, abbr: Optional[str], name: Optional[str],
                         position_type: Optional[str], value: Optional[float] = None, group_code: Optional[str] = None) -> StatCategory:
    obj = session.query(StatCategory).filter_by(league_id=league_id, stat_id=stat_id).one_or_none()
    if obj is None:
        obj = StatCategory(league_id=league_id, stat_id=stat_id, abbr=abbr, name=name,
                           position_type=position_type, value=value, group_code=group_code)
        session.add(obj)
        session.flush()
    else:
        changed = False
        if abbr and obj.abbr != abbr:
            obj.abbr = abbr; changed = True
        if name and obj.name != name:
            obj.name = name; changed = True
        if position_type and obj.position_type != position_type:
            obj.position_type = position_type; changed = True
        
        # Only update value if the new value is not None
        if value is not None:
            v1 = float(obj.value) if obj.value is not None else None
            v2 = float(value)
            if v1 != v2:
                obj.value = value; changed = True
        
        # Only update group_code if the new value is not None
        if group_code is not None and obj.group_code != group_code:
            obj.group_code = group_code; changed = True
            
        if changed:
            session.flush()
    return obj


def upsert_roster_requirement(session: Session, *, league_id: int, position: str, count: int) -> RosterRequirement:
    obj = session.query(RosterRequirement).filter_by(league_id=league_id, position=position).one_or_none()
    if obj is None:
        obj = RosterRequirement(league_id=league_id, position=position, count=count)
        session.add(obj)
        session.flush()
    else:
        if obj.count != count:
            obj.count = count
            session.flush()
    return obj


def upsert_matchup(session: Session, *, league_id: int, week_num: int, matchup_index: int) -> Matchup:
    obj = session.query(Matchup).filter_by(league_id=league_id, week_num=week_num, matchup_index=matchup_index).one_or_none()
    if obj is None:
        obj = Matchup(league_id=league_id, week_num=week_num, matchup_index=matchup_index)
        session.add(obj)
        session.flush()
    return obj


def upsert_weekly_total(session: Session, *, league_id: int, week_num: int, matchup_id: Optional[int], team_key: str,
                        stat_id: int, stat_abbr: Optional[str], value: Any) -> WeeklyTotal:
    # Safely convert value to float or None
    safe_value = None
    if value is not None and str(value).strip() != "":
        try:
            safe_value = float(value)
        except (ValueError, TypeError):
            safe_value = None

    obj = session.query(WeeklyTotal).filter_by(league_id=league_id, week_num=week_num, team_key=team_key, stat_id=stat_id).one_or_none()
    if obj is None:
        obj = WeeklyTotal(league_id=league_id, week_num=week_num, matchup_id=matchup_id, team_key=team_key,
                          stat_id=stat_id, stat_abbr=stat_abbr, value=safe_value)
        session.add(obj)
        session.flush()
    else:
        changed = False
        if obj.matchup_id != matchup_id:
            obj.matchup_id = matchup_id; changed = True
        if obj.stat_abbr != stat_abbr:
            obj.stat_abbr = stat_abbr; changed = True
        
        # Numeric comparison
        curr_val = float(obj.value) if obj.value is not None else None
        if curr_val != safe_value:
            obj.value = safe_value; changed = True
            
        if changed:
            session.flush()
    return obj


def upsert_weekly_player_gp(session: Session, *, league_id: int, week_num: int, team_key: str,
                            player_key: str, gp: int, source: Optional[str] = None,
                            player_name: Optional[str] = None, positions: Optional[str] = None) -> WeeklyPlayerGP:
    # ensure team, player rows exist (team must already be created by caller)
    upsert_player(session, player_key=player_key, name=player_name, positions=positions)
    if positions:
        set_player_positions(session, player_key=player_key, positions=[p.strip() for p in str(positions).split(',') if p.strip()])
    obj = session.query(WeeklyPlayerGP).filter_by(league_id=league_id, week_num=week_num, team_key=team_key, player_key=player_key).one_or_none()
    if obj is None:
        obj = WeeklyPlayerGP(league_id=league_id, week_num=week_num, team_key=team_key, player_key=player_key,
                             gp=int(gp or 0), source=source, computed_at=_dt.datetime.utcnow())
        session.add(obj)
        session.flush()
    else:
        changed = False
        if int(obj.gp or 0) != int(gp or 0):
            obj.gp = int(gp or 0); changed = True
        if source and obj.source != source:
            obj.source = source; changed = True
        if changed:
            obj.computed_at = _dt.datetime.utcnow()
            session.flush()
    return obj


def upsert_roster_slot_daily(session: Session, *, date: _dt.date, league_id: int, team_key: str,
                             player_key: str, selected_position: Optional[str], had_game: Optional[bool],
                             gp: int, player_name: Optional[str] = None, positions: Optional[str] = None) -> RosterSlotDaily:
    """Insert/update a daily roster slot row for a player.

    Uniqueness is defined by (date, league_id, team_key, player_key).
    Ensures the Player exists; Team is expected to already exist (created elsewhere).
    """
    # Ensure player exists and positions are current (best-effort; ignore errors)
    try:
        upsert_player(session, player_key=player_key, name=player_name, positions=positions)
        if positions:
            set_player_positions(session, player_key=player_key,
                                 positions=[p.strip() for p in str(positions).split(',') if p.strip()])
    except Exception:
        # Non-fatal: FK may already be satisfied
        pass

    obj = (
        session.query(RosterSlotDaily)
        .filter_by(date=date, league_id=league_id, team_key=team_key, player_key=player_key)
        .one_or_none()
    )
    if obj is None:
        obj = RosterSlotDaily(
            date=date,
            league_id=league_id,
            team_key=team_key,
            player_key=player_key,
            selected_position=(selected_position or None),
            had_game=had_game,
            gp=int(gp or 0),
        )
        session.add(obj)
        session.flush()
    else:
        changed = False
        sel = (selected_position or None)
        if obj.selected_position != sel:
            obj.selected_position = sel; changed = True
        if obj.had_game != had_game:
            obj.had_game = had_game; changed = True
        if int(obj.gp or 0) != int(gp or 0):
            obj.gp = int(gp or 0); changed = True
        if changed:
            session.flush()
    return obj
