from typing import Dict, Optional

import pandas as pd
from rapidfuzz import process, fuzz

from .base import BaseRegistry, TeamEntry


class TeamRegistry(BaseRegistry):
    def __init__(self, excel_path: str = 'Team2TM.xlsx', sheet: str = 'Team2TM'):
        super().__init__()
        self.excel_path = excel_path
        self.sheet = sheet
        self._teams: Dict[str, TeamEntry] = {}
        self._alias_to_code: Dict[str, str] = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        df = pd.read_excel(self.excel_path, sheet_name=self.sheet)
        if 'TM' not in df.columns:
            raise ValueError("Team2TM.xlsx must include a 'TM' column with canonical team codes")
        self._teams.clear(); self._alias_to_code.clear()
        for _, row in df.iterrows():
            code = str(row['TM']).strip().upper()
            entry = TeamEntry(code=code, name=None, source_map={})
            for col in df.columns:
                if col == 'TM':
                    continue
                val = str(row[col]).strip()
                if val and val.lower() != 'nan':
                    entry.source_map[col] = val
                    self._alias_to_code[val.strip().upper()] = code
            self._teams[code] = entry
        # Also map codes themselves
        for code in self._teams.keys():
            self._alias_to_code[code.upper()] = code
        self._loaded = True

    def to_code(self, s: Optional[str], source_col: Optional[str] = None) -> Optional[str]:
        self.load()
        if not s:
            return None
        key = s.strip().upper()
        if key in self._alias_to_code:
            return self._alias_to_code[key]
        # Fuzzy fallback
        choices = list(self._alias_to_code.keys())
        if not choices:
            return None
        match = process.extractOne(key, choices, scorer=fuzz.WRatio)
        if match:
            mval, score, _ = match
            if score >= 90:
                return self._alias_to_code.get(mval)
        return None

    def list_codes(self):
        self.load()
        return sorted(self._teams.keys())
