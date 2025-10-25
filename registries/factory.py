from typing import Iterable, Dict, Protocol

from .names import NamesRegistry
from .positions import PositionRegistry
from .teams import TeamRegistry
from .reporting import SummaryReporter


class PlayerSourceAdapter(Protocol):
    def fetch(self) -> Iterable[Dict]:
        """Yield dicts: {'display_name': str, 'team': str, 'position': str, 'source': str}"""
        ...


class RegistryFactory:
    def __init__(self):
        self.names = NamesRegistry()
        self.teams = TeamRegistry()
        self.positions = PositionRegistry()

    def update_from_source(self, adapter: PlayerSourceAdapter, reporter: SummaryReporter):
        for row in adapter.fetch():
            self.names.upsert_from_source(
                display_name=row['display_name'], team_raw=row['team'], pos_raw=row['position'],
                source=row.get('source', 'UNKNOWN'), team_registry=self.teams, pos_registry=self.positions,
                reporter=reporter
            )
        self.names.save()
        reporter.save()
