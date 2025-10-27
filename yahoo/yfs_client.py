from typing import Any, Dict, Optional

from yahoo_oauth import OAuth2
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

API_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"


def _make_soup(raw_xml: str):
    """
    Build a BeautifulSoup object trying multiple XML-capable parsers, with graceful fallback.
    Returns a soup or None if all parsers fail.
    """
    for feature in ("lxml-xml", "xml", "html.parser"):
        try:
            return BeautifulSoup(raw_xml, feature)
        except Exception:
            continue
    return None


class YahooFantasyClient:
    def __init__(self, oauth: OAuth2):
        self.oauth = oauth

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = dict(params or {})
        # Prefer XML for consistent parsing with BeautifulSoup
        params.setdefault("format", "xml")
        url = f"{API_BASE}{path}"
        headers = {"Accept": "application/xml, text/xml;q=0.9, application/json;q=0.8"}
        resp = self.oauth.session.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "").lower()
        text = resp.text
        # Always return a unified envelope; include JSON only if provided
        if "json" in ctype:
            try:
                data = resp.json()
                data.setdefault("_source_format", "json")
                data.setdefault("_raw_xml", text if "xml" in text[:100].lower() else None)
                return data
            except Exception:
                pass
        # XML primary path
        parsed = {"_source_format": "xml", "_raw_xml": text}
        return parsed

    # High-level helpers
    def get_games(self, sport: str, season: str | int) -> Dict[str, Any]:
        # Use game_codes filter (sport) and seasons filter.
        path = "/games;game_codes={};seasons={}".format(sport, season)
        return self.get(path)

    def get_game(self, sport: str, season: str | int) -> Dict[str, Any]:
        return self.get_games(sport, season)

    def get_league(self, league_key: str) -> Dict[str, Any]:
        path = f"/league/{league_key}"
        return self.get(path)

    def get_user_leagues_for_game(self, game_key: str) -> Dict[str, Any]:
        # List leagues for the logged-in user within a game
        # Endpoint pattern: /users;use_login=1/games;game_keys=XXX/leagues
        path = f"/users;use_login=1/games;game_keys={game_key}/leagues"
        return self.get(path)

    def get_team_roster(self, team_key: str) -> Dict[str, Any]:
        # Fetch the roster for a specific team key
        # Endpoint pattern: /team/{team_key}/roster
        path = f"/team/{team_key}/roster"
        return self.get(path)


# Convenience function to extract the numeric game_key from the Yahoo response
# Works with either Yahoo JSON (preferred) or XML fallback stored under _raw_xml.

def extract_game_key(games_payload: Dict[str, Any]) -> Optional[str]:
    # Prefer XML parsing first
    raw_xml = None
    if isinstance(games_payload, dict):
        raw_xml = games_payload.get("_raw_xml")
    if raw_xml:
        # 1) Try ElementTree (namespace-agnostic)
        try:
            root = ET.fromstring(raw_xml)
            for elem in root.iter():
                tag = elem.tag
                # strip namespace if present: {ns}local
                if isinstance(tag, str):
                    if '}' in tag:
                        tag = tag.split('}', 1)[1]
                    if tag.lower() == 'game_key':
                        text = (elem.text or '').strip()
                        if text:
                            return text
        except Exception:
            pass
        # 2) Try BeautifulSoup with multiple XML parsers
        for feature in ("lxml-xml", "xml", "html.parser"):
            try:
                soup = BeautifulSoup(raw_xml, feature)
                gk_tag = soup.find(lambda t: hasattr(t, 'name') and t.name and str(t.name).lower().endswith("game_key"))
                if gk_tag and getattr(gk_tag, 'text', None):
                    text = gk_tag.text.strip()
                    if text:
                        return text
            except Exception:
                continue

    # Fallback to JSON-like structure if available
    try:
        fc = games_payload["fantasy_content"]
        games = fc["games"]
        # games can be a dict with numeric keys or a list; normalize iteration
        candidates: list[dict] = []
        if isinstance(games, dict):
            for k, v in games.items():
                if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
                    # typical: {"0": {"game": {...}}} or {"0": {"game": [{...}]}}
                    g = v.get("game")
                    if isinstance(g, list):
                        for item in g:
                            if isinstance(item, dict):
                                candidates.append(item)
                    elif isinstance(g, dict):
                        candidates.append(g)
                # sometimes a direct list or dict under "game"
                if k == "game":
                    g = v
                    if isinstance(g, list):
                        for item in g:
                            if isinstance(item, dict):
                                candidates.append(item)
                    elif isinstance(g, dict):
                        candidates.append(g)
        elif isinstance(games, list):
            for item in games:
                if isinstance(item, dict):
                    g = item.get("game") or item
                    if isinstance(g, dict):
                        candidates.append(g)
        for g in candidates:
            gk = g.get("game_key")
            if gk is not None:
                return str(gk)
    except Exception:
        pass

    return None


def make_league_key(game_key: str, league_id: str | int) -> str:
    return f"{game_key}.l.{league_id}"


def make_team_key(game_key: str, league_id: str | int, team_id: str | int) -> str:
    return f"{game_key}.l.{league_id}.t.{team_id}"


def extract_num_teams(league_payload: Dict[str, Any]) -> Optional[int]:
    """
    Try to extract the number of teams from a league payload (XML or JSON envelope)
    Returns an int or None if not found.
    """
    # XML path
    raw_xml = None
    if isinstance(league_payload, dict):
        raw_xml = league_payload.get("_raw_xml")
    if raw_xml:
        # Try ElementTree first
        try:
            root = ET.fromstring(raw_xml)
            for elem in root.iter():
                tag = elem.tag
                if isinstance(tag, str) and '}' in tag:
                    tag = tag.split('}', 1)[1]
                if isinstance(tag, str) and tag.lower() == 'num_teams':
                    text = (elem.text or '').strip()
                    if text.isdigit():
                        return int(text)
        except Exception:
            pass
        # Try BeautifulSoup
        soup = _make_soup(raw_xml)
        if soup is not None:
            nt = soup.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith('num_teams'))
            if nt and getattr(nt, 'text', None):
                text = nt.text.strip()
                try:
                    return int(text)
                except Exception:
                    pass
    # JSON path
    try:
        fc = league_payload.get('fantasy_content') if isinstance(league_payload, dict) else None
        league = None
        if isinstance(fc, dict):
            # common structures: {"league": {...}} or numeric keys
            if 'league' in fc and isinstance(fc['league'], dict):
                league = fc['league']
            else:
                for k, v in fc.items():
                    if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
                        if 'league' in v and isinstance(v['league'], dict):
                            league = v['league']
                            break
        if isinstance(league, dict):
            # direct num_teams
            nt = league.get('num_teams')
            if isinstance(nt, (int, float)):
                return int(nt)
            if isinstance(nt, str) and nt.isdigit():
                return int(nt)
            # sometimes values are nested as list of dicts
            for key, val in league.items():
                if key == 'num_teams':
                    if isinstance(val, dict) and 'count' in val:
                        try:
                            return int(val['count'])
                        except Exception:
                            pass
        # Some payloads may have settings: {"settings": {"num_teams": 12}}
        settings = (league.get('settings') if isinstance(league, dict) else None)
        if isinstance(settings, dict):
            nt = settings.get('num_teams')
            if isinstance(nt, (int, float)):
                return int(nt)
            if isinstance(nt, str) and nt.isdigit():
                return int(nt)
    except Exception:
        pass
    return None


def parse_roster_xml(raw_xml: str) -> Dict[str, Any]:
    """
    Parse Yahoo roster XML into a simple dict with team metadata and a players list.
    Returns: { team_name, team_key, count, players: [ { player_id, name, positions: [...], selected_position } ] }
    """
    # Try BeautifulSoup with robust parser selection
    soup = _make_soup(raw_xml)
    result: Dict[str, Any] = {"team_name": None, "team_key": None, "count": 0, "players": []}

    if soup is not None:
        # Try to locate the team node
        team_tag = soup.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("team"))
        if team_tag:
            # team_key
            tk = team_tag.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("team_key"))
            if tk and tk.text:
                result["team_key"] = tk.text.strip()
            # team name
            tn = team_tag.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("name"))
            if tn and tn.text:
                result["team_name"] = tn.text.strip()
        # Players
        player_tags = soup.find_all(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("player"))
        players = []
        for p in player_tags:
            pid_tag = p.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("player_id"))
            name_full = None
            name_container = p.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("name"))
            if name_container:
                nf = name_container.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("full"))
                if nf and nf.text:
                    name_full = nf.text.strip()
            if not name_full:
                # fallback: any tag ending with 'name_full' or 'name'
                nf = p.find(lambda t: hasattr(t, 'name') and t.name and (t.name.lower().endswith("name_full") or t.name.lower() == "name"))
                if nf and nf.text:
                    name_full = nf.text.strip()
            # Eligible positions
            positions = []
            elig = p.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("eligible_positions"))
            if elig:
                for pos_tag in elig.find_all(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("position")):
                    if pos_tag.text:
                        positions.append(pos_tag.text.strip())
            # Selected position (if present on roster)
            sel_pos_tag = p.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("selected_position"))
            selected_position = None
            if sel_pos_tag:
                sp = sel_pos_tag.find(lambda t: hasattr(t, 'name') and t.name and t.name.lower().endswith("position"))
                if sp and sp.text:
                    selected_position = sp.text.strip()
            players.append({
                "player_id": pid_tag.text.strip() if pid_tag and pid_tag.text else None,
                "name": name_full,
                "positions": positions,
                "selected_position": selected_position,
            })
        result["players"] = players
        result["count"] = len(players)
        return result

    # Fallback: ElementTree parsing (namespace-agnostic)
    try:
        root = ET.fromstring(raw_xml)
    except Exception:
        return result

    def strip(tag: str) -> str:
        if not isinstance(tag, str):
            return ""
        return tag.split('}', 1)[1] if '}' in tag else tag

    # Find team info
    team_elem = None
    for elem in root.iter():
        if strip(elem.tag).lower() == 'team':
            team_elem = elem
            break
    if team_elem is not None:
        for child in team_elem.iter():
            name = strip(child.tag).lower()
            if name == 'team_key' and child.text:
                result['team_key'] = child.text.strip()
            elif name == 'name' and child.text and not result.get('team_name'):
                result['team_name'] = child.text.strip()

    # Players
    players: list[Dict[str, Any]] = []
    for p in root.iter():
        if strip(p.tag).lower() == 'player':
            player_id = None
            name_full = None
            positions: list[str] = []
            selected_position = None
            for c in p.iter():
                nm = strip(c.tag).lower()
                if nm == 'player_id' and c.text:
                    player_id = c.text.strip()
                elif nm == 'full' and c.text:
                    name_full = c.text.strip()
                elif nm == 'name_full' and c.text and not name_full:
                    name_full = c.text.strip()
                elif nm == 'position' and c.text:
                    # Distinguish eligible vs selected by parent chain when possible
                    parent_nm = strip(getattr(getattr(c, 'getparent', lambda: None)(), 'tag', '')) if hasattr(c, 'getparent') else ''
                    text_val = c.text.strip()
                    # If under selected_position, set selected_position; else add to eligible
                    if parent_nm.lower().endswith('selected_position'):
                        selected_position = text_val
                    else:
                        positions.append(text_val)
            players.append({
                'player_id': player_id,
                'name': name_full,
                'positions': positions,
                'selected_position': selected_position,
            })
    result['players'] = players
    result['count'] = len(players)
    return result
