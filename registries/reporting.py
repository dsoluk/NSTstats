import json
from datetime import datetime

from .base import PlayerRecord


class SummaryReporter:
    def __init__(self, path: str = 'summary.json'):
        self.path = path
        self._data = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'new_players': [],
            'team_changes': [],
            'unmatched': [],
        }

    def log_new_player(self, rec: PlayerRecord):
        self._data['new_players'].append({
            'id': rec.id,
            'display_name': rec.display_name,
            'cname': rec.cname,
            'team': rec.team,
            'position': rec.position,
            'source': rec.source,
        })

    def log_team_change(self, rec: PlayerRecord, old_team: str, new_team: str):
        self._data['team_changes'].append({
            'id': rec.id,
            'display_name': rec.display_name,
            'cname': rec.cname,
            'position': rec.position,
            'old_team': old_team,
            'new_team': new_team,
        })

    def log_unmatched(self, row):
        self._data['unmatched'].append(row)

    def save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
