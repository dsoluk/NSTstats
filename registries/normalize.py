import re
from typing import Dict
from unidecode import unidecode

# Basic nickname/alias map; extend as needed or move to configurable JSON
NICKNAME_MAP: Dict[str, str] = {
    'mitch': 'mitchell',
    'mike': 'michael',
    'matt': 'matthew',
    'andy': 'andrew',
    'alex': 'alexander',
    'chris': 'christopher',
    'will': 'william',
    'bill': 'william',
    'zach': 'zachary',
    'zack': 'zachary',
}


def _clean_token(tok: str) -> str:
    tok = tok.replace('.', '')
    tok = re.sub(r"[^a-z\-']", '', tok)
    return tok


def normalize_name(display_name: str) -> str:
    if not display_name:
        return ''
    s = unidecode(display_name).lower().strip()
    tokens = [t for t in re.split(r"\s+", s) if t]
    if not tokens:
        return ''
    first = tokens[0]
    first = NICKNAME_MAP.get(first, first)

    last_parts = tokens[1:]

    if not last_parts:
        last = ''
    elif len(last_parts) == 1:
        last = _clean_token(last_parts[0])
    else:
        last = '-'.join(_clean_token(t) for t in last_parts if t)

    first = _clean_token(first)

    cname = f"{first} {last}".strip()
    return cname
