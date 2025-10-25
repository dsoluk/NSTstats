from typing import Dict, Optional

from .base import BaseRegistry, PositionEntry


class PositionRegistry(BaseRegistry):
    def __init__(self):
        super().__init__()
        self._positions: Dict[str, PositionEntry] = {
            'F': PositionEntry('F', None),
            'C': PositionEntry('C', 'F'),
            'L': PositionEntry('L', 'F'),
            'R': PositionEntry('R', 'F'),
            'D': PositionEntry('D', None),
            'G': PositionEntry('G', None),
        }
        self._aliases = {
            'C': 'C', 'CEN': 'C', 'CENTER': 'C', 'CENTRE': 'C',
            'LW': 'L', 'L': 'L', 'LEFT': 'L', 'LEFT WING': 'L',
            'RW': 'R', 'R': 'R', 'RIGHT': 'R', 'RIGHT WING': 'R',
            'D': 'D', 'DEF': 'D', 'DEFENSE': 'D', 'DEFENCE': 'D',
            'G': 'G', 'GOL': 'G', 'GOALIE': 'G', 'GK': 'G',
        }

    def normalize(self, raw: Optional[str]) -> Optional[str]:
        code = (raw or '').strip().upper()
        return self._aliases.get(code, code) if code else None

    def get_parent(self, code: str) -> Optional[str]:
        entry = self._positions.get(code)
        return entry.parent if entry else None
