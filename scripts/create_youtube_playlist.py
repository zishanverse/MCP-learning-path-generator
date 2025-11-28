import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import extract_video_ids_from_text  # noqa: E402

load_dotenv()

CLIENT_SECRETS = os.getenv("YOUTUBE_OAUTH_CLIENT_SECRETS_FILE", "youtube_client_secrets.json")
TOKEN_FILE = os.getenv("YOUTUBE_OAUTH_TOKEN_FILE", "tokens/youtube_token.json")


def build_youtube_client():
    """Create an authenticated YouTube Data API client using stored OAuth tokens."""
    if not os.path.exists(CLIENT_SECRETS):
        print(f"Client secrets not found at {CLIENT_SECRETS}. Set YOUTUBE_OAUTH_CLIENT_SECRETS_FILE in .env or place file there.")
        sys.exit(1)

    if not os.path.exists(TOKEN_FILE):
        print(f"Token file not found at {TOKEN_FILE}. Run the OAuth helper first: python scripts/get_youtube_token.py")
        sys.exit(1)

    with open(TOKEN_FILE, "r", encoding="utf-8") as fh:
        token_data = json.load(fh)

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")

    with open(CLIENT_SECRETS, "r", encoding="utf-8") as fh:
        secret_json = json.load(fh)

    client_info = secret_json.get("installed") or secret_json.get("web")
    client_id = client_info.get("client_id")
    client_secret = client_info.get("client_secret")
    token_uri = client_info.get("token_uri", "https://oauth2.googleapis.com/token")

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except Exception:
        print("Missing google API libraries. Run: pip install -r requirements.txt")
        raise

    creds = None
    if access_token or refresh_token:
        creds = Credentials(token=access_token, refresh_token=refresh_token, client_id=client_id, client_secret=client_secret, token_uri=token_uri)

    if creds and (not creds.valid and creds.refresh_token):
        try:
            creds.refresh(Request())
            os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
            with open(TOKEN_FILE, "w", encoding="utf-8") as fh:
                json.dump({"access_token": creds.token, "refresh_token": creds.refresh_token}, fh)
            print("Refreshed access token and updated token file.")
        except Exception as exc:
            print(f"Failed to refresh token: {exc}")

    if not creds or not creds.valid:
        print("Credentials are not valid. Run the OAuth helper first: python scripts/get_youtube_token.py")
        sys.exit(1)

    youtube = build("youtube", "v3", credentials=creds)
    return youtube


def collect_video_ids(explicit_ids: List[str], plan_texts: List[str]) -> List[str]:
    """Combine explicit IDs with those parsed from plan text, preserving order and uniqueness."""
    vids: List[str] = []
    seen: set[str] = set()

    for vid in explicit_ids:
        video_id = vid.strip()
        if video_id and video_id not in seen:
            seen.add(video_id)
            vids.append(video_id)

    for text in plan_texts:
        for video_id in extract_video_ids_from_text(text):
            if video_id not in seen:
                seen.add(video_id)
                vids.append(video_id)

    return vids


def prompt_for_manual_ids(existing: List[str]) -> List[str]:
    """Prompt the user for manual video IDs until they enter a blank line."""
    vids = list(existing)
    seen = set(existing)
    print("No video IDs detected automatically. Enter them manually (press Enter with empty input to finish).")
    while True:
        vid = input("Video ID: ").strip()
        if not vid:
            break
        if vid not in seen:
            seen.add(vid)
            vids.append(vid)
    return vids


def main():
    parser = argparse.ArgumentParser(description="Create a YouTube playlist and optionally add videos.")
    parser.add_argument("--title", type=str, help="Playlist title")
    parser.add_argument("--desc", type=str, help="Playlist description")
    parser.add_argument("--videos", type=str, help="Comma-separated YouTube video IDs to add to the playlist (no spaces)")
    parser.add_argument("--plan-file", type=str, help="Path to a generated learning-path document. Video links/IDs will be parsed automatically.")
    parser.add_argument("--plan-text", type=str, help="Raw learning-path text (if not using a file).")
    args = parser.parse_args()

    youtube = build_youtube_client()

    title = args.title or input("Playlist title: ") or "My Learning Path Playlist"
    desc = args.desc or input("Playlist description (optional): ") or "Created by MCP Learning Path Generator"

    explicit_ids = [v.strip() for v in (args.videos.split(",") if args.videos else []) if v.strip()]

    plan_sources: List[str] = []
    if args.plan_file:
        plan_path = Path(args.plan_file).expanduser().resolve()
        if not plan_path.exists():
            print(f"Plan file not found at {plan_path}")
        else:
            plan_sources.append(plan_path.read_text(encoding="utf-8"))

    if args.plan_text:
        plan_sources.append(args.plan_text)

    vids = collect_video_ids(explicit_ids, plan_sources)

    if not vids:
        if plan_sources or explicit_ids:
            vids = prompt_for_manual_ids(vids)
        else:
            print("No videos provided. Enter at least one video ID.")
            vids = prompt_for_manual_ids(vids)

    if not vids:
        print("No videos available to add. Exiting without creating a playlist.")
        sys.exit(0)

    print("Creating playlist...")
    resp = youtube.playlists().insert(part="snippet,status", body={
        "snippet": {"title": title, "description": desc},
        "status": {"privacyStatus": "public"}
    }).execute()

    playlist_id = resp.get("id")
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    print(f"Created playlist id: {playlist_id}")
    print(f"Playlist URL: {playlist_url}")

    for vid in vids:
        try:
            youtube.playlistItems().insert(part="snippet", body={
                "snippet": {"playlistId": playlist_id, "resourceId": {"kind": "youtube#video", "videoId": vid}}
            }).execute()
            print(f"Added {vid}")
        except Exception as exc:
            print(f"Failed to add {vid}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
