import pandas as pd


def _coerce_to_numeric(series: pd.Series) -> pd.Series:
    return


def _coerce_toi_to_timedelta(series: pd.Series) -> pd.Series:
    # Try to parse MM:SS first; non-strings will become NaT
    td = pd.to_timedelta(series, errors="coerce")
    # For remaining NaT, try interpreting numeric minutes → Timedelta
    mask = td.isna()
    if mask.any():
        as_num = pd.to_numeric(series[mask], errors="coerce")
        td.loc[mask] = pd.to_timedelta(as_num, unit="m", errors="coerce")
    return td


def basic_cleansing(souptable):
    headers = [th.get_text(strip=True) for th in souptable.find("tr").find_all("th")]
    rows = []
    for row in souptable.find_all("tr")[1:]:
        cells = row.find_all("td")
        cleaned = []
        for cell in cells:
            for a in cell.find_all("a"):
                a.unwrap()
            cleaned.append(cell.get_text(strip=True))
        rows.append(cleaned)

    df = pd.DataFrame(rows, columns=headers)

    for col in df.columns:
        # Coerce TOI/GP into Timedelta regardless of input format
        if "toi" in col.lower():
            df[col] = _coerce_toi_to_timedelta(df[col])
        elif col.lower() in ["player", "team", "position"]:
            df[col] = df[col].astype("string")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="integer")

    # print(f"Col dtypes {df.dtypes}")
    return df


def basic_filtering(df: pd.DataFrame):
    unnamed_cols = [col for col in df.columns if not col.strip()]
    if unnamed_cols:
        # print(f"Dropping unnamed columns: {unnamed_cols}")
        df.drop(columns=unnamed_cols, inplace=True)

    if "TOI/GP" in df.columns:
        toi = _coerce_toi_to_timedelta(df["TOI/GP"])  # idempotent if already timedelta
        df = df[toi >= pd.Timedelta(minutes=1)]
    return df


