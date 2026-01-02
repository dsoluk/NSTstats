import os
from dotenv import load_dotenv
import json

# Try to load Django settings to use Parameter model if available
try:
    import django
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'nststats_project.settings')
    django.setup()
    from nst_app.models import Parameter
    HAS_DJANGO = True
except Exception:
    HAS_DJANGO = False


def get_param(name, default=None):
    if HAS_DJANGO:
        try:
            return Parameter.objects.get(key=name).value
        except Parameter.DoesNotExist:
            pass
    return os.getenv(name, default)


def load_default_params():
    load_dotenv()
    params = {
        "fromseason": get_param("FROMSEASON"),
        "thruseason": get_param("THRUSEASON"),
        "stype": get_param("STYPE", "2"),
        "sit": get_param("SIT"),
        "score": get_param("SCORE", "all"),
        "stdoi": get_param("STDOI", "std"),
        "rate": get_param("RATE", "y"),
        "team": get_param("TEAM", "ALL"),
        "pos": get_param("POS", "S"),
        "loc": get_param("LOC", "B"),
        "toi": get_param("TOI"),
        "gpfilt": get_param("GPFILT", "gpteam"),
        "fd": get_param("FD"),
        "td": get_param("TD"),
        "tgp": get_param("TGP"),
        "lines": get_param("LINES"),
        "draftteam": get_param("DRAFTTEAM")
    }
    url = get_param("BASE_URL")

    # weights = {
    #     "offensive": get_param("WEIGHTS_OFFENSIVE"),
    #     "defensive": get_param("WEIGHTS_DEFENSIVE"),
    #     "special": get_param("WEIGHTS_SPECIAL")
    # }

    # Load columns configuration. It can be provided either as a dict mapping
    # original NST column -> short name (preferred), or as a list of columns
    # plus a separate COLUMN_RENAMES mapping (legacy). We normalize to dicts.
    def _load_json_env(name, default="{}"):
        raw = get_param(name)
        if raw is None:
            raw = default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"Warning: {name} in .env or database is not valid JSON. Using default.")
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
        env_index_weights_raw = get_param("INDEX_WEIGHTS")
        if env_index_weights_raw:
            env_index_weights = json.loads(env_index_weights_raw)
        else:
            env_index_weights = {}
        index_weights = _deep_merge(default_index_weights, env_index_weights)
    except Exception:
        index_weights = default_index_weights

    return params, url, index_weights, std_columns, ppp_columns, goalie_columns
