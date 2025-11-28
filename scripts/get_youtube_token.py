import os
import json
from dotenv import load_dotenv

load_dotenv()

def _resolve_path(path: str | None) -> str | None:
    if not path:
        return path
    return os.path.abspath(os.path.expanduser(path))


def obtain_youtube_token(client_secrets: str | None = None, token_file: str | None = None, scopes: list | None = None, port: int | None = None) -> dict:
    """Run InstalledAppFlow to obtain YouTube OAuth tokens and save them to token_file.

    Returns a dict with access_token and refresh_token.
    """
    CLIENT_SECRETS = _resolve_path(client_secrets or os.getenv("YOUTUBE_OAUTH_CLIENT_SECRETS_FILE", "youtube_client_secrets.json"))
    TOKEN_FILE = _resolve_path(token_file or os.getenv("YOUTUBE_OAUTH_TOKEN_FILE", "tokens/youtube_token.json"))
    SCOPES = scopes or os.getenv(
        "YOUTUBE_OAUTH_SCOPES",
        "https://www.googleapis.com/auth/youtube.force-ssl https://www.googleapis.com/auth/youtube"
    ).split()

    PORT = int(port or os.getenv("YOUTUBE_OAUTH_LOCAL_PORT", "8501"))

    if not os.path.exists(CLIENT_SECRETS):
        raise FileNotFoundError(f"Client secrets file not found at {CLIENT_SECRETS}. Place your client_secrets.json there or update .env.")

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except Exception as e:
        raise RuntimeError("Missing google-auth-oauthlib. Run: pip install google-auth-oauthlib") from e

    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS, scopes=SCOPES)
    print(f"Starting local OAuth server on http://localhost:{PORT}/ to receive callback. If you registered a different redirect URI, set YOUTUBE_OAUTH_LOCAL_PORT accordingly.")
    try:
        creds = flow.run_local_server(open_browser=True, port=PORT)
    except OSError as err:
        if getattr(err, "winerror", None) == 10048:
            raise RuntimeError(
                f"Port {PORT} is in use (likely Streamlit). Stop the app or set YOUTUBE_OAUTH_LOCAL_PORT to an unused port (e.g., 8765) and add http://localhost:{PORT}/ to your Google OAuth redirect URIs."
            ) from err
        raise

    token = getattr(creds, "token", None)
    refresh = getattr(creds, "refresh_token", None)
    data = {"access_token": token, "refresh_token": refresh}

    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh)

    print(f"Saved token file to {TOKEN_FILE}")
    if token:
        print(f"Access token (truncated): {token[:80]}...")
    else:
        print("No access token returned by the OAuth flow.")
        if refresh:
            print("A refresh token was returned — the token can be exchanged later to obtain an access token.")
        else:
            print("No refresh token was returned. If you need a refresh token, ensure the OAuth client is a Desktop type or re-create credentials and choose 'consent' to force a refresh token.")

    return data


if __name__ == "__main__":
    # Allow running as a script for interactive use
    try:
        data = obtain_youtube_token()
        print("OAuth flow completed.")
    except Exception as e:
        print(f"OAuth helper failed: {e}")
        raise
