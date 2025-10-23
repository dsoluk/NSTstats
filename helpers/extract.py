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

            # Find the first table and print its first 5 rows to console, for debugging.
            # TODO can use .prettify from soup?
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")
                for row in rows[:5]:  # Limit to first 5 rows
                    cols = row.find_all(["td", "th"])
                    print([col.text.strip() for col in cols])
            else:
                print("No table found on the page.")
            return table
        else:
            print(f"NST request failed: HTTP {resp.status_code}")
            return None
    except Exception as e:
        print(f"NST request error: {e}")
        return None
