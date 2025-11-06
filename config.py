import os
from dotenv import load_dotenv
import json


def load_default_params():
    load_dotenv()
    params = {
        "fromseason": os.getenv("FROMSEASON"),
        "thruseason": os.getenv("THRUSEASON"),
        "stype": os.getenv("STYPE", "2"),
        "sit": os.getenv("SIT"),
        "score": os.getenv("SCORE", "all"),
        "stdoi": os.getenv("STDOI", "std"),
        "rate": os.getenv("RATE", "y"),
        "team": os.getenv("TEAM", "ALL"),
        "pos": os.getenv("POS", "S"),
        "loc": os.getenv("LOC", "B"),
        "toi": os.getenv("TOI"),
        "gpfilt": os.getenv("GPFILT", "gpteam"),
        "fd": os.getenv("FD"),
        "td": os.getenv("TD"),
        "tgp": os.getenv("TGP"),
        "lines": os.getenv("LINES"),
        "draftteam": os.getenv("DRAFTTEAM")
    }
    url = os.getenv("BASE_URL")

    # weights = {
    #     "offensive": os.getenv("WEIGHTS_OFFENSIVE"),
    #     "defensive": os.getenv("WEIGHTS_DEFENSIVE"),
    #     "special": os.getenv("WEIGHTS_SPECIAL")
    # }

    # Load columns configuration. It can be provided either as a dict mapping
    # original NST column -> short name (preferred), or as a list of columns
    # plus a separate COLUMN_RENAMES mapping (legacy). We normalize to dicts.
    def _load_json_env(name, default="{}"):
        raw = os.getenv(name)
        if raw is None:
            raw = default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"Warning: {name} in .env is not valid JSON. Using default.")
            return json.loads(default)

    std_cfg = _load_json_env("STD_COLUMNS", default="{}")
    ppp_cfg = _load_json_env("PPP_COLUMNS", default="{}")
    goalie_cfg = _load_json_env("GOALIE_COLUMNS", default="{}")

    def _normalize(cfg):
        # If cfg is a list, convert to dict using legacy_map, mapping each name
        # to its short name if available, else keep original (identity).
        # if isinstance(cfg, list):
        #     return {col: legacy_map.get(col, col) for col in cfg}
        # If already a dict, assume proper mapping
        if isinstance(cfg, dict):
            return cfg
        # Unknown type -> fallback empty dict
        return {}

    std_columns = _normalize(std_cfg)
    ppp_columns = _normalize(ppp_cfg)
    goalie_columns = _normalize(goalie_cfg)

    # Index weights configuration with sensible defaults. Can be overridden via INDEX_WEIGHTS env var (JSON).
    def _deep_merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out

    default_index_weights = {
        "offensive": {  # per-stat weights for offensive index
            "G": 1.0,
            "A": 1.0,
            "PPP": 1.0,
            "SOG": 1.0,
            "FOW": 1.0,
        },
        "banger": {     # per-stat weights for banger index
            "HIT": 1.0,
            "BLK": 1.0,
            "PIM": 1.0,
        },
        "composite": {  # weights for combining off/banger into composite
            "offensive": 0.7,
            "banger": 0.3,
        }
    }

    # Allow overriding via env var as JSON
    try:
        env_index_weights = json.loads(os.getenv("INDEX_WEIGHTS", "{}"))
        index_weights = _deep_merge(default_index_weights, env_index_weights)
    except Exception:
        index_weights = default_index_weights

    return params, url, index_weights, std_columns, ppp_columns, goalie_columns
