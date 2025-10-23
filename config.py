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

    weights = {
        "offensive": os.getenv("WEIGHTS_OFFENSIVE"),
        "defensive": os.getenv("WEIGHTS_DEFENSIVE"),
        "special": os.getenv("WEIGHTS_SPECIAL")
    }

    selected_columns = json.loads(os.getenv("SELECTED_COLUMNS"))
    goalie_columns = json.loads(os.getenv("GOALIE_COLUMNS"))

    # rename map
    raw_map = os.getenv("COLUMN_RENAMES", "{}")
    try:
        column_renames = json.loads(raw_map)
    except json.JSONDecodeError:
        print("Warning: COLUMN_RENAMES in .env is not valid JSON. Ignoring.")
        column_renames = {}

    return params, url, weights, selected_columns, goalie_columns, column_renames
