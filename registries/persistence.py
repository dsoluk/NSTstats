import csv
from typing import Iterable

from .base import PlayerRecord


def export_players_csv(records: Iterable[PlayerRecord], path: str = 'player_registry.csv'):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'cname', 'display_name', 'team', 'position', 'source'])
        for r in records:
            writer.writerow([r.id, r.cname, r.display_name, r.team, r.position, r.source])
