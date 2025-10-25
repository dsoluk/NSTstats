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

    # Provide a simple default weights structure to preserve the function signature
    weights = {
        "offensive": {},
        "defensive": {},
        "special": {}
    }
    return params, url, weights, std_columns, ppp_columns, goalie_columns
