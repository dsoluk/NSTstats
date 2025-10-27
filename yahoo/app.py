import json
import os
import sys
import csv
from typing import Optional

from dotenv import load_dotenv

from yahoo.yahoo_auth import get_oauth
from yahoo.yfs_client import (
    YahooFantasyClient,
    extract_game_key,
    make_league_key,
    make_team_key,
    parse_roster_xml,
    extract_num_teams,
)


class YahooFantasyApp:
    """
    Object-oriented orchestrator for the Yahoo Fantasy helper.

    Responsibilities:
    - Load configuration from environment (.env supported)
    - Initialize OAuth and API client
    - Execute the main workflow (fetch game, league, roster)
    """

    def __init__(self,
                 sport: Optional[str] = None,
                 season: Optional[str] = None,
                 league_id: Optional[str] = None,
                 status: Optional[str] = None,
                 count: Optional[str] = None,
                 position: Optional[str] = None,
                 team_id: Optional[str] = None):
        # Allow explicit params to override environment
        load_dotenv()
        self.sport = sport or self._env("SPORT", "nhl")
        self.season = season or self._env("SEASON", "2025")
        self.league_id = league_id or self._env("LEAGUE_ID")
        self.status = status or self._env("STATUS", "")
        self.count = count or self._env("COUNT", "")
        self.position = position or self._env("POSITION")
        self.team_id = team_id or self._env("TEAM_ID", "4")

        client_id = self._env("YAHOO_CLIENT_ID")
        client_secret = self._env("YAHOO_CLIENT_SECRET")

        try:
            oauth = get_oauth(client_id, client_secret)
        except Exception as e:
            print(f"OAuth error: {e}")
            sys.exit(1)

        self.client = YahooFantasyClient(oauth)

    @staticmethod
    def _env(name: str, default: Optional[str] = None) -> Optional[str]:
        v = os.environ.get(name)
        return v if v is not None and v != "" else default

    @staticmethod
    def _pretty(obj) -> str:
        return json.dumps(obj, indent=2, ensure_ascii=False)

    def run(self) -> None:
        print("Starting Yahoo Fantasy NHL helper (OO mode)...")
        print(
            f"Config: SPORT={self.sport} SEASON={self.season} LEAGUE_ID={self.league_id} "
            f"STATUS={self.status} COUNT={self.count} POSITION={self.position}"
        )

        # 1) Get game object for sport+season
        print(f"Fetching game for sport={self.sport} season={self.season} ...")
        game_json = self.client.get_game(self.sport, self.season)
        src_fmt = game_json.get("_source_format") if isinstance(game_json, dict) else None
        if src_fmt:
            print(f"\n[Info] Response source format: {src_fmt}")
            if src_fmt == "xml":
                print("[Info] Yahoo returned XML. Using BeautifulSoup fallback to extract values.")
        print("\n=== Game Payload ===")
        print(self._pretty(game_json))

        game_key = extract_game_key(game_json)
        if not game_key:
            print("\nCould not extract game_key from response.")
            if src_fmt == "xml":
                print("Hint: Yahoo responded with XML. Ensure the endpoint and parameters are correct. We'll continue improving XML parsing if needed.")
            sys.exit(2)

        # 2) If LEAGUE_ID set, fetch that league. Otherwise, list user's leagues for the game.
        if self.league_id:
            league_key = make_league_key(game_key, self.league_id)
            print(f"\nFetching league {league_key} ...")
            league_json = self.client.get_league(league_key)
            print("\n=== League Payload ===")
            print(self._pretty(league_json))

            # Fetch roster for a specific team (retain existing behavior)
            try:
                team_key = make_team_key(game_key, self.league_id, self.team_id)
                print(f"\nFetching roster for team {team_key} ...")
                roster_payload = self.client.get_team_roster(team_key)
                src_fmt_roster = roster_payload.get("_source_format") if isinstance(roster_payload, dict) else None
                raw_xml = roster_payload.get("_raw_xml") if isinstance(roster_payload, dict) else None
                if src_fmt_roster == "xml" and raw_xml:
                    roster = parse_roster_xml(raw_xml)
                    print(f"Roster players count: {roster.get('count')}\nFirst 3 players: {self._pretty(roster.get('players', [])[:3])}")
                    # Save roster JSON to data/
                    out_dir = os.path.join("data")
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"roster_team{self.team_id}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(self._pretty(roster))
                    print(f"Saved roster to {out_path}")
                else:
                    print("Roster response was not XML as expected; printing raw payload for inspection.")
                    print(self._pretty(roster_payload))
            except Exception as e:
                print(f"Error fetching/parsing roster: {e}")

            # Build a single CSV across all teams in the league
            try:
                num_teams = extract_num_teams(league_json)
                # Allow override via env NUM_TEAMS if parsing fails
                if not num_teams:
                    env_nt = self._env("NUM_TEAMS")
                    if env_nt and str(env_nt).isdigit():
                        num_teams = int(env_nt)
                if not num_teams:
                    num_teams = 12  # sensible default per request
                print(f"\nBuilding combined roster CSV for {num_teams} teams...")
                rows = []
                for i in range(1, int(num_teams) + 1):
                    try:
                        tkey = make_team_key(game_key, self.league_id, str(i))
                        print(f"Fetching roster for team_id={i} ({tkey}) ...")
                        payload = self.client.get_team_roster(tkey)
                        if isinstance(payload, dict) and payload.get("_source_format") == "xml" and payload.get("_raw_xml"):
                            r = parse_roster_xml(payload["_raw_xml"])
                            team_name = r.get("team_name")
                            team_key = r.get("team_key") or tkey
                            for p in r.get("players", []) or []:
                                positions = p.get("positions") or []
                                rows.append({
                                    "team_id": i,
                                    "team_key": team_key,
                                    "team_name": team_name,
                                    "player_id": p.get("player_id"),
                                    "name": p.get("name"),
                                    "positions": ";".join(positions) if isinstance(positions, list) else positions,
                                    "selected_position": p.get("selected_position"),
                                })
                        else:
                            print(f"[Warn] Unexpected roster payload for team {i}; skipping or inspect manually.")
                    except Exception as inner_e:
                        print(f"[Warn] Failed to fetch/parse roster for team {i}: {inner_e}")
                        continue
                # Write CSV
                out_dir = os.path.join("data")
                os.makedirs(out_dir, exist_ok=True)
                csv_path = os.path.join(out_dir, "all_rosters.csv")
                fieldnames = ["team_id", "team_key", "team_name", "player_id", "name", "positions", "selected_position"]
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                print(f"Saved combined roster CSV to {csv_path} with {len(rows)} rows.")
            except Exception as e:
                print(f"Error building combined roster CSV: {e}")
        else:
            print("\nLEAGUE_ID not set. Listing your leagues for this game so you can pick one...")
            leagues_json = self.client.get_user_leagues_for_game(game_key)
            print("\n=== Your Leagues for this game ===")
            print(self._pretty(leagues_json))
            print("\nTip: Find your league_id in the printed JSON and set LEAGUE_ID in .env, then re-run.")
