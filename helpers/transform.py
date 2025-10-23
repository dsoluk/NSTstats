import pandas as pd

from datetime import timedelta

def _parse_toi(toi_str):
    """Convert MM:SS string to timedelta"""
    try:
        minutes, seconds = map(int, toi_str.split(":"))
        return timedelta(minutes=minutes, seconds=seconds)
    except:
        return pd.NaT  # or timedelta(0) if you prefer



def basic_cleansing(souptable):

    # Step 1: Extract headers
    headers = [th.get_text(strip=True) for th in souptable.find("tr").find_all("th")]

    # Step 2: Extract rows and clean <a> tags
    data_rows = []
    for row in souptable.find_all("tr")[1:]:  # Skip header
        cells = row.find_all("td")
        cleaned_cells = []
        for cell in cells:
            # Remove <a> tags but keep their text
            for a in cell.find_all("a"):
                a.unwrap()
            cleaned_cells.append(cell.get_text(strip=True))
        data_rows.append(cleaned_cells)

    # Step 3: Convert to DataFrame
    df = pd.DataFrame(data_rows, columns=headers)

    # Step 4: Clean and convert decimal columns
    # for col in df.columns:
    #     if "/60" or "/gp" in col.lower():
    #         df[col] = pd.to_numeric(df[col], errors="coerce")
            # df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Step 5: Convert TOI column to timedelta
    for col in df.columns:
        if "toi/gp" in col.lower():
            df[col] = df[col].apply(_parse_toi)
            # df[col] = df[col].fillna(timedelta(0))

    # Step 6: Clean and convert integer columns
    # int_cols = ["GP", "G", "A", "P", "Shots"]
    # for col in int_cols:
    #     if col in df.columns:
    #         df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def basic_filtering(df: pd.DataFrame):
    # find unnamed columns and remove them
    unnamed_cols = [col for col in df.columns if not col.strip()]
    if unnamed_cols:
        print(f"Dropping unnamed columns: {unnamed_cols}")
        df.drop(columns=unnamed_cols, inplace=True)

    # Filter rows where TOI/GP < 10 minutes
    if "TOI/GP" in df.columns:
        df = df[df["TOI/GP"]  >= 10]

    return df


