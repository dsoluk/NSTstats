import itertools
import json
from typing import Dict, Tuple, List, Optional

from .base import BaseRegistry, PlayerRecord
from .normalize import normalize_name
from .positions import PositionRegistry
from .teams import TeamRegistry


class NamesRegistry(BaseRegistry):
    def __init__(self, json_path: str = 'player_registry.json'):
        super().__init__()
        self.json_path = json_path
        self._records: Dict[int, PlayerRecord] = {}
        self._by_key: Dict[Tuple[str, str, str], int] = {}
        self._next_id = itertools.count(1)
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            self._loaded = True
            return
        max_id = 0
        for row in data:
            rec = PlayerRecord(**row)
            self._records[rec.id] = rec
            self._by_key[(rec.cname, rec.team, rec.position)] = rec.id
            if rec.id > max_id:
                max_id = rec.id
        self._next_id = itertools.count(max_id + 1)
        self._loaded = True

    def save(self):
        self.load()
        payload = [
            {
                'id': r.id,
                'cname': r.cname,
                'display_name': r.display_name,
                'team': r.team,
                'position': r.position,
                'source': r.source,
                'extra': r.extra,
            } for r in self._records.values()
        ]
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def find(self, cname: str, team: str, position: str) -> Optional[PlayerRecord]:
        self.load()
        rid = self._by_key.get((cname, team, position))
        return self._records.get(rid) if rid else None

    def upsert_from_source(self, display_name: str, team_raw: str, pos_raw: str, source: str,
                           team_registry: TeamRegistry, pos_registry: PositionRegistry,
                           reporter=None) -> PlayerRecord:
        self.load()
        cname = normalize_name(display_name)
        team = team_registry.to_code(team_raw)
        pos = pos_registry.normalize(pos_raw)
        if not team or not pos:
            raise ValueError(f"Unable to normalize team '{team_raw}' or position '{pos_raw}' for {display_name}")

        rec = self.find(cname, team, pos)
        if rec:
            return rec

        # Check same cname different team/position
        candidates = [r for r in self._records.values() if r.cname == cname]
        if candidates:
            for cand in candidates:
                if cand.position == pos and cand.team != team:
                    if reporter:
                        reporter.log_team_change(cand, old_team=cand.team, new_team=team)
                    updated = PlayerRecord(
                        id=cand.id, cname=cand.cname, display_name=display_name, team=team,
                        position=pos, source=source, extra=cand.extra
                    )
                    self._records[cand.id] = updated
                    del self._by_key[(cand.cname, cand.team, cand.position)]
                    self._by_key[(updated.cname, updated.team, updated.position)] = updated.id
                    return updated
        # Insert new
        new_id = next(self._next_id)
        new_rec = PlayerRecord(
            id=new_id, cname=cname, display_name=display_name, team=team, position=pos, source=source
        )
        self._records[new_id] = new_rec
        self._by_key[(cname, team, pos)] = new_id
        if reporter:
            reporter.log_new_player(new_rec)
        return new_rec

    def all(self) -> List[PlayerRecord]:
        self.load()
        return list(self._records.values())
