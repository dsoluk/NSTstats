from typing import Optional

from yahoo_oauth import OAuth2


def get_oauth(client_id: Optional[str], client_secret: Optional[str]) -> OAuth2:
    """
    Initialize Yahoo OAuth2 using yahoo_oauth. First run will open a browser to authorize.
    The library will cache tokens in tokens.json in the working directory.
    """
    if not client_id or not client_secret:
        raise RuntimeError(
            "YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET are required. Set them in your .env file."
        )
    oauth = OAuth2(client_id, client_secret)
    if not oauth.token_is_valid():
        oauth.refresh_access_token()
    return oauth


def _get(oauth: OAuth2, url: str) -> dict:
    resp = oauth.session.get(url)
    resp.raise_for_status()
    return resp.json()
