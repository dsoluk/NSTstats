import functools
import json
import os
import re
import unicodedata
from typing import Optional, Dict

import pandas as pd

_TEAM_MAP_CACHE: Optional[Dict[str, str]] = None


def _load_team_map_from_excel(xlsx_path: str = 'Team2TM.xlsx') -> Dict[str, str]:
    global _TEAM_MAP_CACHE
    if _TEAM_MAP_CACHE is not None:
        return _TEAM_MAP_CACHE
    mapping: Dict[str, str] = {}
    try:
        if os.path.exists(xlsx_path):
            df = pd.read_excel(xlsx_path)
            # Expect columns like: Team, Code (or similar). Be permissive.
            cols = {c.lower(): c for c in df.columns}
            team_col = cols.get('team') or cols.get('name') or list(df.columns)[0]
            code_col = cols.get('code') or cols.get('tm') or cols.get('abbr') or list(df.columns)[-1]
            for _, row in df.iterrows():
                raw = str(row[team_col]).strip()
                code = str(row[code_col]).strip().upper()
                if not raw or not code:
                    continue
                mapping[raw.upper()] = code
        # Common aliases to codes (uppercase keys)
        aliases = {
            'ARI': 'ARI', 'PHX': 'ARI', 'ARZ': 'ARI',
            'LA': 'LAK', 'LAK': 'LAK', 'L.A.': 'LAK', 'LOS ANGELES': 'LAK',
            'NJD': 'NJD', 'NJ': 'NJD', 'N.J.': 'NJD', 'NEW JERSEY': 'NJD',
            'NYI': 'NYI', 'NY ISLANDERS': 'NYI', 'ISLANDERS': 'NYI',
            'NYR': 'NYR', 'NY RANGERS': 'NYR', 'RANGERS': 'NYR',
            'TBL': 'TBL', 'TB': 'TBL', 'TAMPA BAY': 'TBL',
            'VGK': 'VGK', 'VEGAS': 'VGK',
            'WSH': 'WSH', 'WAS': 'WSH', 'WASHINGTON': 'WSH',
            'WPG': 'WPG', 'WINNIPEG': 'WPG',
            'MTL': 'MTL', 'MON': 'MTL', 'MONTREAL': 'MTL',
            'FLA': 'FLA', 'FLORIDA': 'FLA',
            'CAR': 'CAR', 'HURRICANES': 'CAR',
            'BOS': 'BOS', 'BRUINS': 'BOS',
            'BUF': 'BUF', 'SABRES': 'BUF',
            'CGY': 'CGY', 'CALGARY': 'CGY',
            'CHI': 'CHI', 'BLACKHAWKS': 'CHI',
            'CBJ': 'CBJ', 'CLB': 'CBJ', 'COLUMBUS': 'CBJ',
            'COL': 'COL', 'AVALANCHE': 'COL',
            'DAL': 'DAL', 'STARS': 'DAL',
            'DET': 'DET', 'RED WINGS': 'DET',
            'EDM': 'EDM', 'OILERS': 'EDM',
            'MIN': 'MIN', 'WILD': 'MIN',
            'NSH': 'NSH', 'NASHVILLE': 'NSH',
            'NHL': 'NHL',
            'N/A': '', '': '', 'FA': '', 'FREE AGENT': '',
            'SEA': 'SEA', 'SEATTLE': 'SEA',
            'SJS': 'SJS', 'SJ': 'SJS', 'SAN JOSE': 'SJS',
            'STL': 'STL', 'BLUES': 'STL',
            'VAN': 'VAN', 'CANUCKS': 'VAN',
            'BOS.': 'BOS', 'MTL.': 'MTL',
            'ANA': 'ANA', 'DUCKS': 'ANA', 'ANAHEIM': 'ANA',
            'OTT': 'OTT', 'SENATORS': 'OTT',
            'PHI': 'PHI', 'PHILADELPHIA': 'PHI',
            'PIT': 'PIT', 'PENGUINS': 'PIT',
            'TOR': 'TOR', 'MAPLE LEAFS': 'TOR',
            'ARI COYOTES': 'ARI', 'COYOTES': 'ARI',
        }
        mapping.update(aliases)
    except Exception:
        # If Excel read fails, rely on aliases only
        mapping = aliases.copy()  # type: ignore[name-defined]
    _TEAM_MAP_CACHE = mapping
    return mapping


def to_team_code(team: str) -> str:
    t = str(team or '').strip().upper()
    if not t:
        return ''
    mapping = _load_team_map_from_excel()
    # Try exact
    code = mapping.get(t)
    if code:
        return code
    # Remove punctuation and extra spaces
    t2 = re.sub(r"[^A-Z]", "", t)
    for k, v in mapping.items():
        if re.sub(r"[^A-Z]", "", k) == t2:
            return v
    # Last chance: already looks like a code (3 letters)
    if len(t) == 3 and t.isalpha():
        return t
    return ''


_SUFFIXES = [
    'jr', 'sr', 'ii', 'iii', 'iv', 'v',
]


# common nickname/alternate name mapping (normalized keys)
_NICKNAME_MAP = {
    'alexander': 'alex',
    'mitchell': 'mitch',
    'nicholas': 'nick',
    'christopher': 'chris',
    'matthew': 'matt',
    'zachary': 'zach',
    'zack': 'zach',
    'zackary': 'zach',
    'jacob': 'jake',
    'anthony': 'tony',
    'william': 'will',
    'robert': 'bob',
    'michael': 'mike',
    'vince': 'vincent',
    'evgenii': 'evgeny',
    'alexei': 'alexey',
    'sergei': 'sergey',
}


def normalize_name(name: str) -> str:
    s = str(name or '').strip()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = s.replace('-', ' ')
    s = re.sub(r"[^a-z' ]+", ' ', s)
    s = re.sub(r"\s+", ' ', s).strip()
    # remove suffixes at end
    parts = s.split(' ')
    if parts and parts[-1] in _SUFFIXES:
        parts = parts[:-1]
    # special: remove dots from initials like "t.j." -> "tj"
    parts = [p.replace("'", '') for p in parts]
    
    # Apply nickname mapping to first name if it matches
    if parts:
        first = parts[0]
        if first in _NICKNAME_MAP:
            parts[0] = _NICKNAME_MAP[first]

    return ' '.join(parts)


def normalize_position(pos: str) -> str:
    p = str(pos or '').strip().upper()
    if not p:
        return ''
    # Map common labels to target codes
    forward = {'F', 'FW', 'FORWARD'}
    centers = {'C', 'CEN', 'CTR'}
    left = {'LW', 'L', 'LWF', 'L W'}
    right = {'RW', 'R', 'RWF', 'R W'}
    defense = {'D', 'DEF', 'LD', 'RD', 'D/LD', 'D/RD'}
    goalie = {'G', 'GOL', 'GOLIE', 'GOALIE'}
    bench_like = {'UTIL', 'BN', 'IR', 'IR+', 'NA'}
    if p in goalie:
        return 'G'
    if p in defense:
        return 'D'
    if p in centers:
        return 'C'
    if p in left:
        return 'L'
    if p in right:
        return 'R'
    if p in forward:
        # If only generic forward, pick C as default to keep joins stable
        return 'C'
    return ''


def to_fpos(pos_code: str) -> str:
    y = str(pos_code or '').upper()
    if y in {'C', 'L', 'R'}:
        return 'F'
    if y in {'D', 'G'}:
        return y
    return ''
