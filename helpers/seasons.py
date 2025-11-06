def previous_season_label(label: str) -> str:
    """
    Given an NST season label like "20252026", return the previous season label "20242025".
    Falls back to returning the input on parse errors.
    """
    try:
        s = str(label).strip()
        if len(s) != 8 or not s.isdigit():
            return s
        start = int(s[:4])
        end = int(s[4:])
        return f"{start-1:04d}{end-1:04d}"
    except Exception:
        return label
