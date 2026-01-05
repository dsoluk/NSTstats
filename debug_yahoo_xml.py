
import os
from yahoo.yfs_client import parse_players_xml
from yahoo.app import YahooFantasyApp

def debug_fetch():
    app = YahooFantasyApp()
    client = app.client
    league_key = os.getenv("YAHOO_LEAGUE_KEY", "465.l.83511")
    
    print(f"Fetching players for league {league_key}...")
    payload = client.get_league_players(league_key, position="G", count=5)
    raw_xml = payload.get("_raw_xml")
    if not raw_xml:
        print("No XML returned")
        return

    print("RAW XML (first 1000 chars):")
    print(raw_xml[:1000])
    
    players = parse_players_xml(raw_xml)
    print("\nParsed Players:")
    for p in players:
        print(p)

if __name__ == "__main__":
    debug_fetch()
