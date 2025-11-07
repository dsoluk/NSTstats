from typing import Iterable, Dict

import requests
from bs4 import BeautifulSoup


class NSTPlayerTeamsAdapter:
    """
    Lightweight scraper for NST playerteams.php, which lists players with team and position.
    This prefers NST as the source of truth for current team/position. It applies no normalization here;
    downstream code should normalize via helpers.normalization.
    """
    BASE_URL = 'https://www.naturalstattrick.com/playerteams.php'

    def __init__(self, headers=None, timeout=30):
        self.headers = headers or {'User-Agent': 'Mozilla/5.0 (compatible; NSTstats/1.0)'}
        self.timeout = timeout

    def fetch(self) -> Iterable[Dict]:
        r = requests.get(self.BASE_URL, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        rows = []
        tables = soup.find_all('table')
        for tbl in tables:
            headers = [th.get_text(strip=True).lower() for th in tbl.find_all('th')]
            if not headers:
                continue
            header_text = "|".join(headers)
            # Require these columns to be present in some form
            if not all(h in header_text for h in ['player', 'team', 'position']):
                continue
            tbody = tbl.find('tbody') or tbl
            for tr in tbody.find_all('tr'):
                tds = [td.get_text(strip=True) for td in tr.find_all('td')]
                if len(tds) < len(headers):
                    continue
                idx_map = {h: i for i, h in enumerate(headers)}
                name = tds[idx_map.get('player', 0)]
                team = tds[idx_map.get('team', 1)]
                pos = tds[idx_map.get('position', 2)]
                if name and team and pos:
                    rows.append({'display_name': name, 'team': team, 'position': pos, 'source': 'NST'})
        return rows
