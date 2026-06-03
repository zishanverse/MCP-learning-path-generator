"""
utils.py — Planning utilities for the Learning Path Generator.

This module contains pure text-processing and read-only helper functions,
such as extracting YouTube video IDs from markdown and checking their availability.

It does NOT perform any external writes or actions on behalf of the user.
"""
from __future__ import annotations

import os
import re
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, Tuple, Set, List

import httpx
from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Return absolute path for a possibly relative/env path."""
    if not path:
        return path
    return os.path.abspath(os.path.expanduser(path))


def _build_public_youtube_client():
    """Build a YouTube client using the GOOGLE_API_KEY for read-only public data.
    
    This does not require OAuth and avoids using developer credentials for actions.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing. Cannot access YouTube Data API.")
    
    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise RuntimeError("Missing google-api-python-client") from e

    return build("youtube", "v3", developerKey=api_key)


def _normalize_video_id(candidate: str) -> Optional[str]:
    """Return a canonical 11-char video id when possible."""
    if not candidate:
        return None
    candidate = candidate.strip()
    match = re.match(r"^[A-Za-z0-9_-]{11}$", candidate)
    if match:
        return candidate
    match = re.match(r"^([A-Za-z0-9_-]{11})", candidate)
    if match:
        return match.group(1)
    return None


def collect_video_title_candidates(text: str) -> List[str]:
    """Return possible video titles gleaned from structured resource lines."""
    if not text:
        return []

    candidates: List[str] = []

    def _push(value: Optional[str]):
        if not value:
            return
        cleaned = value.strip().strip("-–:•·")
        if len(cleaned) >= 5:
            candidates.append(cleaned)

    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if "type" in lower and "video" in lower:
            quoted = re.findall(r"\"([^\"\r\n]{5,200})\"", stripped)
            if quoted:
                for q in quoted:
                    _push(q)
            else:
                before = re.split(r"[-–]\s*type", stripped, flags=re.IGNORECASE)[0]
                before = re.sub(r"^[\-*•\d\.\)\s]+", "", before)
                _push(before)
        if "youtube" in lower and "http" not in lower:
            quoted = re.findall(r"\"([^\"\r\n]{5,200})\"", stripped)
            if quoted:
                for q in quoted:
                    _push(q)
            else:
                segment = stripped.split("YouTube", 1)[0]
                segment = re.sub(r"^[\-*•\d\.\)\s]+", "", segment)
                _push(segment)

    bracketed = re.findall(r"\[(.{5,200}?)\]", text)
    for token in bracketed:
        tok = token.strip()
        if tok and not re.match(r"PL[a-zA-Z0-9_-]+", tok):
            candidates.append(tok)

    deduped: List[str] = []
    seen: Set[str] = set()
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            deduped.append(cand)
    return deduped


def search_youtube_for_title(
    title: str,
    prefer_recent: bool = True,
    exclude_ids: Optional[Set[str]] = None
) -> Optional[str]:
    """Search YouTube for a title and return the best video ID or None."""
    if not _bool_env("ENABLE_YOUTUBE_TITLE_LOOKUP", False):
        return None
    try:
        youtube = _build_public_youtube_client()
        exclude = set(exclude_ids or [])
        cutoff = datetime.utcnow() - timedelta(days=730)

        def _run_query(recent: bool) -> List[dict]:
            params = {
                "part": "snippet",
                "q": title,
                "type": "video",
                "maxResults": 10,
            }
            if recent:
                params["order"] = "date"
                params["publishedAfter"] = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                params["order"] = "relevance"
            resp = youtube.search().list(**params).execute()
            return resp.get("items", [])

        search_order = [True, False] if prefer_recent else [False]
        for recent in search_order:
            items = _run_query(recent)
            for item in items:
                vid = item.get("id", {}).get("videoId")
                if not vid or vid in exclude:
                    continue
                if recent:
                    published_at = item.get("snippet", {}).get("publishedAt")
                    if published_at:
                        try:
                            published_dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                            if published_dt < cutoff:
                                continue
                        except Exception:
                            pass
                return vid
        return None
    except Exception as e:
        print(f"[utils] search_youtube_for_title failed: {e}")
        return None


def extract_video_ids_from_text(
    text: str,
    fetch_generic_urls: bool = True,
    max_fetches: int = 5,
    title_hints: Optional[List[str]] = None
) -> List[str]:
    """Extract unique YouTube video IDs from arbitrary text, fetching generic links if needed."""
    ids: List[str] = []
    seen: Set[str] = set()
    if not text:
        return ids

    def _record(video_id: Optional[str]):
        if video_id and video_id not in seen:
            seen.add(video_id)
            ids.append(video_id)

    direct_patterns = [
        r"(?:youtube\.com/(?:watch\?v=|embed/|v/|shorts/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"watch\?v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"video\s*id[:\-\s]+([A-Za-z0-9_-]{11})",
    ]

    for pattern in direct_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            _record(_normalize_video_id(match.group(1)))

    url_pattern = re.compile(r"https?://[^\s)]+", flags=re.IGNORECASE)
    fetch_targets: List[str] = []
    for match in url_pattern.finditer(text):
        url = match.group(0)
        if "youtu" not in url.lower():
            continue
        parsed = urllib.parse.urlparse(url)
        host = parsed.netloc.lower()
        if "youtube.com" not in host and "youtu.be" not in host:
            continue
        query = urllib.parse.parse_qs(parsed.query)
        vid = None
        if "v" in query:
            vid = _normalize_video_id(query["v"][0])
        else:
            path_parts = [segment for segment in parsed.path.split("/") if segment]
            if host.endswith("youtu.be") and path_parts:
                vid = _normalize_video_id(path_parts[0])
            elif host.endswith("youtube.com") and path_parts:
                if path_parts[0] in {"embed", "v", "shorts", "live"} and len(path_parts) > 1:
                    vid = _normalize_video_id(path_parts[1])
        if vid:
            _record(vid)
        else:
            fetch_targets.append(url)

    if fetch_generic_urls and fetch_targets:
        client = httpx.Client(timeout=10.0, follow_redirects=True)
        try:
            for url in fetch_targets[:max_fetches]:
                try:
                    resp = client.get(url)
                    html = resp.text
                except Exception:
                    continue
                match = re.search(r"watch\?v=([A-Za-z0-9_-]{11})", html)
                if match:
                    _record(_normalize_video_id(match.group(1)))
        finally:
            try:
                client.close()
            except Exception:
                pass

    title_candidates = list(title_hints or [])
    for candidate in collect_video_title_candidates(text):
        if candidate not in title_candidates:
            title_candidates.append(candidate)

    if title_candidates:
        for title in title_candidates:
            if len(ids) >= 40:
                break
            vid = search_youtube_for_title(title, exclude_ids=seen)
            _record(_normalize_video_id(vid) if vid else None)

    return ids


def filter_available_videos(video_ids: List[str]) -> Tuple[List[str], List[str]]:
    """Return (available, unavailable) video IDs using YouTube Data API status checks."""
    if not video_ids:
        return [], []
    deduped: List[str] = []
    seen: Set[str] = set()
    for vid in video_ids:
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(vid)
    if not _bool_env("ENABLE_YOUTUBE_AVAILABILITY_CHECK", False):
        return deduped, []

    try:
        youtube = _build_public_youtube_client()
        available: List[str] = []
        unavailable: List[str] = []
        for i in range(0, len(deduped), 50):
            chunk = deduped[i:i+50]
            resp = youtube.videos().list(part="status", id=",".join(chunk)).execute()
            returned = {item.get("id"): item for item in resp.get("items", [])}
            for vid in chunk:
                item = returned.get(vid)
                if not item:
                    unavailable.append(vid)
                    continue
                status = item.get("status", {})
                if status.get("privacyStatus") == "public" and status.get("uploadStatus") == "processed":
                    available.append(vid)
                else:
                    unavailable.append(vid)
        return available, [vid for vid in unavailable if vid not in available]
    except Exception as e:
        print(f"[utils] filter_available_videos failed: {e}")
        return deduped, []