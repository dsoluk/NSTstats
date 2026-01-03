import requests
from bs4 import BeautifulSoup


def fetch_nst_html(url, params=None):
    """Fetch NaturalStatTrick player list page HTML (playerlist.php). Returns text or None on failure.
    Note: Parameters are kept for backward compatibility but not used by playerlist.php.
    """
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            html_content = resp.text

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Find the first table
            table = soup.find("table")
            if not table:
                print(f"[Warn] No table found in NST response for URL: {resp.url}")
            return table
        else:
            print(f"[Warn] NST request failed: HTTP {resp.status_code} for URL: {resp.url}")
            return None
    except Exception as e:
        print(f"[Error] NST request error: {e}")
        return None
