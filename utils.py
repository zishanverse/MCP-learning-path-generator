from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from prompt import user_goal_prompt
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import os
import socket
import urllib.parse
import httpx
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from typing import Optional, Tuple, Any, Callable, Dict, List
import asyncio
import json
from pydantic import BaseModel, Field, field_validator
import atexit
import re

# Global helpers for DirectHTTPTool instances
DIRECT_HTTP_TOOL_INSTANCES: list = []
# Max concurrency for direct HTTP requests (protect against socket exhaustion)
DIRECT_MAX_CONCURRENCY = int(os.getenv("DIRECT_MAX_CONCURRENCY", "5"))
_DIRECT_SEMAPHORE = asyncio.Semaphore(DIRECT_MAX_CONCURRENCY)


def _close_all_direct_clients():
    """Close all AsyncClient instances created for DirectHTTPTool on process exit.

    Uses asyncio.run to close clients cleanly; falls back to sync close if needed.
    """
    async def _close_all_async():
        for t in DIRECT_HTTP_TOOL_INSTANCES:
            try:
                await t._client.aclose()
            except Exception:
                try:
                    # try sync close as fallback
                    t._client.close()
                except Exception:
                    pass

    try:
        asyncio.run(_close_all_async())
    except Exception:
        # best-effort fallback
        for t in DIRECT_HTTP_TOOL_INSTANCES:
            try:
                t._client.close()
            except Exception:
                pass


atexit.register(_close_all_direct_clients)

# HIGHLIGHT: Load environment variables from the .env file
load_dotenv()


def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Return absolute path for a possibly relative/env path."""
    if not path:
        return path
    return os.path.abspath(os.path.expanduser(path))

cfg = RunnableConfig(recursion_limit=100)


def get_youtube_access_token() -> str:
    """Obtain a YouTube OAuth access token.

    Priority:
    1. YOUTUBE_OAUTH_ACCESS_TOKEN env var (manual short-term token)
    2. tokens file at YOUTUBE_OAUTH_TOKEN_FILE (cached)
    3. Run InstalledAppFlow using client secrets file at YOUTUBE_OAUTH_CLIENT_SECRETS_FILE
    """
    # 1. direct env override
    token = os.getenv("YOUTUBE_OAUTH_ACCESS_TOKEN")
    if token:
        return token

    token_file = _resolve_path(os.getenv("YOUTUBE_OAUTH_TOKEN_FILE", "tokens/youtube_token.json"))
    if os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict) and data.get("access_token"):
                    return data.get("access_token")
                if data.get("token"):
                    return data.get("token")
        except Exception:
            pass

    # 3. Run the InstalledAppFlow if client secrets are provided
    client_secrets = _resolve_path(os.getenv("YOUTUBE_OAUTH_CLIENT_SECRETS_FILE"))
    if not client_secrets or not os.path.exists(client_secrets):
        raise RuntimeError(
            "YouTube OAuth client configuration not found. Set YOUTUBE_OAUTH_ACCESS_TOKEN or YOUTUBE_OAUTH_CLIENT_SECRETS_FILE pointing to your OAuth client_secrets.json"
        )

    scopes = os.getenv(
        "YOUTUBE_OAUTH_SCOPES",
        "https://www.googleapis.com/auth/youtube.force-ssl https://www.googleapis.com/auth/youtube"
    ).split()

    try:
        # Local import to avoid hard dependency unless this feature is used
        from google_auth_oauthlib.flow import InstalledAppFlow

        port = int(os.getenv("YOUTUBE_OAUTH_LOCAL_PORT", "8501"))
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets, scopes=scopes)
        print(f"Starting local OAuth server on http://localhost:{port}/ to receive callback")
        creds = flow.run_local_server(open_browser=True, port=port)
        token = getattr(creds, "token", None)

        # Cache token
        try:
            os.makedirs(os.path.dirname(token_file), exist_ok=True)
            with open(token_file, "w", encoding="utf-8") as fh:
                json.dump({"access_token": token, "refresh_token": getattr(creds, "refresh_token", None)}, fh)
        except Exception:
            pass

        if not token:
            raise RuntimeError("Failed to obtain access token from OAuth flow")
        return token
    except Exception as e:
        raise RuntimeError(f"YouTube OAuth flow failed: {e}") from e


def create_youtube_playlist_and_add_videos(video_ids: list[str], title: str = "Learning Path Playlist", description: str = "Created by MCP Learning Path Generator") -> str:
    """Create a YouTube playlist using stored OAuth tokens and add the given video IDs.

    Returns the created playlist ID.
    """
    # Load token and client info
    token_file = _resolve_path(os.getenv("YOUTUBE_OAUTH_TOKEN_FILE", "tokens/youtube_token.json"))
    client_secrets = _resolve_path(os.getenv("YOUTUBE_OAUTH_CLIENT_SECRETS_FILE"))
    if not client_secrets or not os.path.exists(client_secrets):
        raise RuntimeError("OAuth client secrets file not found. Set YOUTUBE_OAUTH_CLIENT_SECRETS_FILE in .env")

    if not os.path.exists(token_file):
        raise RuntimeError(f"Token file not found at {token_file}. Run the OAuth helper first.")

    with open(token_file, "r", encoding="utf-8") as fh:
        token_data = json.load(fh)

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")

    with open(client_secrets, "r", encoding="utf-8") as fh:
        secret_json = json.load(fh)

    client_info = secret_json.get("installed") or secret_json.get("web")
    client_id = client_info.get("client_id")
    client_secret = client_info.get("client_secret")
    token_uri = client_info.get("token_uri", "https://oauth2.googleapis.com/token")

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except Exception as e:
        raise RuntimeError("Missing google API libraries. Install google-api-python-client and google-auth") from e

    creds = Credentials(token=access_token, refresh_token=refresh_token, client_id=client_id, client_secret=client_secret, token_uri=token_uri)
    # Refresh if needed
    try:
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
            # persist refreshed token
            try:
                with open(token_file, "w", encoding="utf-8") as fh:
                    json.dump({"access_token": creds.token, "refresh_token": creds.refresh_token}, fh)
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"Failed to refresh credentials: {e}") from e

    youtube = build("youtube", "v3", credentials=creds)

    # Create playlist
    try:
        resp = youtube.playlists().insert(part="snippet,status", body={
            "snippet": {"title": title, "description": description},
            "status": {"privacyStatus": "public"}
        }).execute()
        playlist_id = resp.get("id")
    except Exception as e:
        raise RuntimeError(f"Failed to create playlist: {e}") from e

    # Add videos
    for vid in video_ids:
        try:
            youtube.playlistItems().insert(part="snippet", body={
                "snippet": {"playlistId": playlist_id, "resourceId": {"kind": "youtube#video", "videoId": vid}}
            }).execute()
        except Exception:
            # continue on failure to add individual videos
            pass

    return playlist_id


def _build_youtube_client_from_token():
    """Build and return a youtube client (googleapiclient) from stored token and client secrets.

    Returns tuple (youtube_client, credentials)
    """
    token_file = _resolve_path(os.getenv("YOUTUBE_OAUTH_TOKEN_FILE", "tokens/youtube_token.json"))
    client_secrets = _resolve_path(os.getenv("YOUTUBE_OAUTH_CLIENT_SECRETS_FILE"))
    if not client_secrets or not os.path.exists(client_secrets):
        raise RuntimeError("OAuth client secrets file not found. Set YOUTUBE_OAUTH_CLIENT_SECRETS_FILE in .env")
    if not os.path.exists(token_file):
        # Attempt to run the helper function in-process to obtain a token interactively
        helper_mod_path = os.path.join(os.path.dirname(__file__), "scripts", "get_youtube_token.py")
        invoked = False
        if os.path.exists(helper_mod_path):
            try:
                # First try importing the scripts module and calling the function
                try:
                    import importlib
                    helper_mod = importlib.import_module("scripts.get_youtube_token")
                    if hasattr(helper_mod, "obtain_youtube_token"):
                        print("Token file not found: invoking in-process OAuth helper...")
                        helper_mod.obtain_youtube_token(client_secrets=None, token_file=token_file, scopes=None, port=None)
                        invoked = True
                except Exception:
                    # Fall back to running the helper as a subprocess
                    import subprocess, sys
                    print("Falling back to running token helper as subprocess...")
                    subprocess.run([sys.executable, helper_mod_path], check=True)
                    invoked = True
            except Exception as e:
                raise RuntimeError(f"Failed to run token helper script: {e}") from e
        if not os.path.exists(token_file):
            raise RuntimeError(f"Token file not found at {token_file} after running helper. Please run scripts/get_youtube_token.py manually.")

    with open(token_file, "r", encoding="utf-8") as fh:
        token_data = json.load(fh)

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")

    with open(client_secrets, "r", encoding="utf-8") as fh:
        secret_json = json.load(fh)

    client_info = secret_json.get("installed") or secret_json.get("web")
    client_id = client_info.get("client_id")
    client_secret = client_info.get("client_secret")
    token_uri = client_info.get("token_uri", "https://oauth2.googleapis.com/token")

    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except Exception as e:
        raise RuntimeError("Missing google API libraries. Install google-api-python-client and google-auth") from e

    creds = Credentials(token=access_token, refresh_token=refresh_token, client_id=client_id, client_secret=client_secret, token_uri=token_uri)
    # Refresh if needed
    try:
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
            # persist refreshed token
            try:
                with open(token_file, "w", encoding="utf-8") as fh:
                    json.dump({"access_token": creds.token, "refresh_token": creds.refresh_token}, fh)
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"Failed to refresh credentials: {e}") from e

    youtube = build("youtube", "v3", credentials=creds)
    return youtube, creds


def get_playlist_item_video_ids(playlist_id: str) -> list[str]:
    """Return list of video IDs already present in the playlist."""
    vids: list[str] = []
    try:
        youtube, _ = _build_youtube_client_from_token()
        req = youtube.playlistItems().list(part="snippet", playlistId=playlist_id, maxResults=50)
        while req:
            resp = req.execute()
            for item in resp.get("items", []):
                vid = item.get("snippet", {}).get("resourceId", {}).get("videoId")
                if vid:
                    vids.append(vid)
            req = youtube.playlistItems().list_next(req, resp)
    except Exception:
        pass
    return vids


def add_videos_to_playlist(playlist_id: str, video_ids: list[str]) -> dict:
    """Add given video IDs to an existing playlist. Returns dict with added and failed lists."""
    added = []
    failed = []
    try:
        youtube, _ = _build_youtube_client_from_token()
        for vid in video_ids:
            try:
                youtube.playlistItems().insert(part="snippet", body={
                    "snippet": {"playlistId": playlist_id, "resourceId": {"kind": "youtube#video", "videoId": vid}}
                }).execute()
                added.append(vid)
            except Exception:
                failed.append(vid)
    except Exception:
        return {"added": added, "failed": video_ids}
    return {"added": added, "failed": failed}


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


def collect_video_title_candidates(text: str) -> list[str]:
    """Return possible video titles gleaned from structured resource lines."""
    if not text:
        return []

    candidates: list[str] = []

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

    deduped: list[str] = []
    seen: set[str] = set()
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            deduped.append(cand)
    return deduped


def extract_video_ids_from_text(
    text: str,
    fetch_generic_urls: bool = True,
    max_fetches: int = 5,
    title_hints: Optional[list[str]] = None
) -> list[str]:
    """Extract unique YouTube video IDs from arbitrary text, fetching generic links if needed."""
    ids: list[str] = []
    seen: set[str] = set()
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
    fetch_targets: list[str] = []
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


def filter_available_videos(video_ids: list[str]) -> tuple[list[str], list[str]]:
    """Return (available, unavailable) video IDs using YouTube Data API status checks."""
    if not video_ids:
        return [], []
    deduped: list[str] = []
    seen: set[str] = set()
    for vid in video_ids:
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(vid)
    try:
        youtube, _ = _build_youtube_client_from_token()
        available: list[str] = []
        unavailable: list[str] = []
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
    except Exception:
        return deduped, []


def search_youtube_for_title(
    title: str,
    prefer_recent: bool = True,
    exclude_ids: Optional[set[str]] = None
) -> Optional[str]:
    """Search YouTube for a title and return the best video ID or None."""
    try:
        youtube, _ = _build_youtube_client_from_token()
        exclude = set(exclude_ids or [])
        cutoff = datetime.utcnow() - timedelta(days=730)

        def _run_query(recent: bool) -> list[dict]:
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
    except Exception:
        return None


def infer_requested_day_count(goal_text: str, fallback: int = 10) -> int:
    """Infer requested day count from user goal text, defaulting when absent."""
    clamp = lambda value: max(1, min(60, value))
    if not goal_text:
        return clamp(fallback)
    text = goal_text.strip().lower()
    # direct "X day(s)" pattern
    match = re.search(r"(\d+)\s*(?:day|days)", text)
    if match:
        return clamp(int(match.group(1)))
    # hyphenated e.g. 5-day
    match = re.search(r"(\d+)[-\s]*(?:day|days)", text)
    if match:
        return clamp(int(match.group(1)))
    # weeks -> convert to days
    match = re.search(r"(\d+)\s*(?:week|weeks)", text)
    if match:
        return clamp(int(match.group(1)) * 7)
    # months -> approx 30days
    match = re.search(r"(\d+)\s*(?:month|months)", text)
    if match:
        return clamp(int(match.group(1)) * 30)
    return clamp(fallback)


def remove_placeholder_lines(text: str) -> str:
    """Strip lines that still contain placeholder markers to avoid blank links."""
    if not text:
        return text
    cleaned_lines = []
    for line in text.splitlines():
        lower = line.lower()
        if "placeholder" in lower:
            continue
        if "insert_youtube_playlist_link" in lower:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

#function to dynamically initialize any supported model
def initialize_model(model_name: str) -> Any:
    if "gemini" in model_name:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    # Claude replacement → Hugging Face LLaMA
    elif "claude" in model_name or "llama" in model_name:
        hf_key = os.getenv("HF_API_KEY")
        if not hf_key:
            raise ValueError("Hugging Face API key not found. Please set HF_API_KEY in your .env file.")
         # 1. Create the basic endpoint
        llm = HuggingFaceEndpoint(
            # repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            repo_id="NousResearch/Meta-Llama-3-8B-Instruct", 
            # task="text-generation",
            max_new_tokens=1024,
            huggingfacehub_api_token=hf_key
        )
        
        # 2. Wrap it in a ChatHuggingFace object
        return ChatHuggingFace(llm=llm)
    # Mistral via Hugging Face
    elif "mistral" in model_name:
        hf_key = os.getenv("HF_API_KEY")
        if not hf_key:
            raise ValueError("Hugging Face API key not found. Please set HF_API_KEY in your .env file.")
        # 1. Create the basic endpoint
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            # task="text-generation",
            max_new_tokens=1024,
            huggingfacehub_api_token=hf_key
        )
        
        # 2. Wrap it in a ChatHuggingFace object
        return ChatHuggingFace(llm=llm)

        

    # Mistral via Hugging Face
    elif "mistral" in model_name:
        hf_key = os.getenv("HF_API_KEY")
        if not hf_key:
            raise ValueError("Hugging Face API key not found. Please set HF_API_KEY in your .env file.")
        # 1. Create the basic endpoint
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            # task="text-generation",
            max_new_tokens=1024,
            huggingfacehub_api_token=hf_key
        )

        # 2. Wrap it in a ChatHuggingFace object
        return ChatHuggingFace(llm=llm)


class ComposioAwareTool(BaseTool):
    """Tool wrapper that handles Composio-specific requirements"""
    
    name: str
    description: str
    original_tool: Any
    connected_account_id: Optional[str] = None
    
    class ComposioInput(BaseModel):
        """Input model that handles common Composio parameters"""
        model_config = {"extra": "allow", "str_strip_whitespace": True}
        
        # Common parameters for different tool types
        q: Optional[str] = Field(None, description="Query parameter")
        query: Optional[str] = Field(None, description="Search query")
        toolkit_name: Optional[str] = Field(None, description="Toolkit name")
        connected_account_id: Optional[str] = Field(None, description="Connected account ID")
        params: Optional[Any] = Field(None, description="Additional parameters (can be dict or string)")
        
        @field_validator('params', mode='before')
        @classmethod
        def parse_params(cls, v):
            """Parse params if it's a JSON string"""
            if isinstance(v, str):
                try:
                    import json
                    return json.loads(v)
                except json.JSONDecodeError:
                    return {"value": v}
            return v
        
        def model_dump(self, **kwargs):
            """Override model_dump to include extra fields"""
            data = super().model_dump(**kwargs)
            # Add any extra fields that were set
            for key, value in self.__dict__.items():
                if key not in data and not key.startswith('_') and value is not None:
                    data[key] = value
            return data
    
    def __init__(self, original_tool: Any, connected_account_id: Optional[str] = None, **kwargs):
        super().__init__(
            name=original_tool.name,
            description=self._create_description(original_tool),
            original_tool=original_tool,
            **kwargs
        )
        self.args_schema = self.ComposioInput
        self.connected_account_id = connected_account_id
    
    def _create_description(self, tool):
        """Create a helpful description based on tool name"""
        name = tool.name.lower()
        base_desc = tool.description or f"Execute {tool.name}"
        
        if 'youtube' in name or 'search' in name:
            return f"{base_desc}. Use 'q' parameter for search query."
        elif 'notion' in name:
            return f"{base_desc}. Notion-related operation."
        elif 'composio' in name:
            if 'search' in name:
                return f"{base_desc}. Search for available tools."
            elif 'execute' in name:
                return f"{base_desc}. Execute a specific tool with toolkit_name and params."
        
        return base_desc
    
    def _prepare_composio_params(self, **kwargs):
        """Prepare parameters specifically for Composio tools"""
        tool_name = self.name.lower()
        cleaned_params = {}
        
        # Handle different tool types
        if 'search_tools' in tool_name:
            # For COMPOSIO_SEARCH_TOOLS - search for YouTube tools
            cleaned_params = {
                "query": kwargs.get('q') or kwargs.get('query') or "youtube"
            }
        elif 'execute_tool' in tool_name:
            # For COMPOSIO_EXECUTE_TOOL - need toolkit_name and params
            query = kwargs.get('q') or kwargs.get('query') or ""
            
            # Get connected account ID from multiple sources
            account_id = (
                kwargs.get('connected_account_id') or 
                self.connected_account_id or
                "default"  # Fallback to default
            )
            
            cleaned_params = {
                "toolkit_name": kwargs.get('toolkit_name') or "youtube",
                "connected_account_id": account_id,
                "params": {"q": query} if query else kwargs.get('params', {})
            }
            
            # Debug logging
            print(f"DEBUG: Prepared execute_tool params: {cleaned_params}")
            
        elif 'youtube' in tool_name or 'search' in tool_name:
            # For direct YouTube tools
            query = kwargs.get('q') or kwargs.get('query') or ""
            if query:
                cleaned_params = {"q": query}
        elif 'notion' in tool_name:
            # For Notion tools - pass through relevant params
            for key, value in kwargs.items():
                if value is not None and key not in ['connected_account_id', 'toolkit_name']:
                    cleaned_params[key] = value
        else:
            # Default: pass through non-None values
            for key, value in kwargs.items():
                if value is not None:
                    cleaned_params[key] = value
        
        return cleaned_params

    def _fallback_youtube_search(self, query: str) -> str:
        """Fallback response for YouTube search when tools fail"""
        if not query:
            return "Please provide a search query for YouTube videos."
        
        return f"""🔧 **YouTube Search Configuration Issue**

I attempted to search YouTube for '{query}' but encountered a configuration issue with the tools.

**Troubleshooting Steps:**
1. ✓ Verify your YouTube integration is properly set up in Composio/Pipedream
2. ✓ Ensure the connected account has proper permissions
3. ✓ Check that your connected_account_id is correctly configured
4. ✓ Try searching manually for '{query}' on YouTube

**Suggested YouTube Search Terms:**
- "{query} tutorial"
- "{query} basics"
- "{query} beginner guide"
- "{query} step by step"

**Manual Search URL**: https://www.youtube.com/results?search_query={query.replace(' ', '+')}

For now, I'll continue with the learning path creation using general recommendations."""

    def _run(self, **kwargs) -> str:
        """Synchronous execution fallback"""
        try:
            return asyncio.run(self._arun(**kwargs))
        except Exception as e:
            error_msg = f"Error executing {self.name}: {str(e)}"
            print(error_msg)
            return error_msg
    
    async def _arun(self, **kwargs) -> Any:
        """Async runner that correctly awaits the underlying/original tool.

        This method normalizes params for Composio-style tools and then
        delegates to the original tool's async/sync interface in a safe way.
        """
        cleaned = self._prepare_composio_params(**kwargs)

        orig = getattr(self, "original_tool", None)
        if orig is None:
            raise RuntimeError("No original tool available to run")

        try:
            # If original tool exposes an async ainvoke, await it
            if hasattr(orig, "ainvoke") and asyncio.iscoroutinefunction(getattr(orig, "ainvoke")):
                result = await orig.ainvoke(cleaned)

            # If original tool exposes an async _arun, await it
            elif hasattr(orig, "_arun") and asyncio.iscoroutinefunction(getattr(orig, "_arun")):
                result = await orig._arun(**cleaned)

            # If original tool is sync (run/_run/callable), run it in threadpool
            else:
                loop = asyncio.get_running_loop()

                def call_sync():
                    # Prefer run or _run if available
                    if hasattr(orig, "run"):
                        try:
                            return orig.run(**cleaned)
                        except TypeError:
                            return orig.run(cleaned)
                    if hasattr(orig, "_run"):
                        try:
                            return orig._run(**cleaned)
                        except TypeError:
                            return orig._run(cleaned)
                    # Fallback to calling the object directly
                    try:
                        return orig(**cleaned)
                    except TypeError:
                        return orig(cleaned)

                result = await loop.run_in_executor(None, call_sync)

        except Exception as e:
            # Return a clear error message so the agent can handle it gracefully
            err = f"Tool '{self.name}' execution failed: {e}"
            print(err)
            # If this is a YouTube-related tool and a query was provided, attempt a graceful fallback:
            try:
                lname = (self.name or "").lower()
                q = cleaned.get("q") or cleaned.get("query") if isinstance(cleaned, dict) else None
                if q and ("youtube" in lname or "search" in lname or "video" in lname):
                    print(f"Attempting fallback YouTube search for query='{q}'")
                    vid = None
                    try:
                        vid = search_youtube_for_title(q)
                    except Exception as e2:
                        print(f"Fallback search failed: {e2}")
                    if vid:
                        # Return a short structured result that the agent can include
                        url = f"https://www.youtube.com/watch?v={vid}"
                        return {"successful": True, "data": {"items": [{"title": q, "url": url, "id": vid}]}, "raw": url}
            except Exception:
                pass
            return err

        # Normalize structured responses from DirectHTTPTool
        if isinstance(result, dict) and "successful" in result:
            # If the request was successful and there is parsed data, prefer it
            if result.get("successful"):
                if result.get("data") is not None:
                    return result["data"]
                if result.get("raw") is not None:
                    return result["raw"]
                return ""  # empty success
            else:
                # propagate a helpful error string
                return f"Remote tool '{self.name}' error: status={result.get('status_code')} body={str(result.get('raw') or result.get('data'))[:400]}"

        # Otherwise return whatever the underlying tool returned
        return result

    async def ainvoke(self, params: Any = None, config: Optional[RunnableConfig] = None):
        """Compatibility method used by agent runtimes that call .ainvoke(payload)."""
        # ToolNode passes a ToolCall dict: {'id', 'name', 'args', 'type': 'tool_call' }
        call_id = None
        kwargs = {}
        if isinstance(params, dict):
            call_id = params.get("id")
            # prefer 'args' (ToolCall schema)
            if "args" in params and isinstance(params["args"], dict):
                kwargs = params["args"]
            # fall back to nested 'params'
            elif "params" in params and isinstance(params["params"], dict):
                kwargs = params["params"]
            else:
                # treat the dict as kwargs directly
                kwargs = params

        else:
            # non-dict payload -> wrap as single param
            kwargs = {"params": params}

        result = await self._arun(**kwargs)

        # If the underlying tool returned the DirectHTTPTool structured dict, handle it
        status = "success"
        content = result
        if isinstance(result, dict) and "successful" in result:
            if not result.get("successful"):
                status = "error"
                content = f"Remote tool '{self.name}' error: status={result.get('status_code')} body={str(result.get('raw') or result.get('data'))[:400]}"
            else:
                # prefer parsed data if available
                content = result.get("data") or result.get("raw") or ""

        # Ensure content is serializable; ToolNode will format further
        return ToolMessage(content=content, name=self.name, tool_call_id=call_id, status=status)

async def setup_agent_with_tools(
    # google_api_key: str,
    use_youtube: bool = True,
    use_drive: bool = False,
    use_notion: bool = False,
    connected_account_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    model_name: str = "gemini-2.5-flash"
) -> Any:
    """
    Set up the agent with Composio hosted MCP server
    """
    try:
        if progress_callback:
            progress_callback("Setting up agent with Composio MCP... ✅")
        
        # Get Composio API key from environment
        composio_api_key = os.getenv("COMPOSIO_API_KEY")
        if not composio_api_key:
            raise ValueError("COMPOSIO_API_KEY not found in environment. Please set it in your .env file.")
        
        # Initialize tools configuration using Composio's hosted MCP server
        tools_config = {}
        
        # Composio hosted MCP server URL (can be overridden)
        # If you have your own MCP-compatible host or per-tool URLs, set the
        # environment variable COMPOSIO_MCP_URL or per-tool overrides as shown below.
        composio_mcp_url = os.getenv("COMPOSIO_MCP_URL", "https://mcp.composio.com")
        
        if use_youtube:
            # Allow per-tool override for the YouTube tool URL (useful when you host
            # the streamable/sse endpoint yourself). If not set, fall back to the
            # MCP host defined in COMPOSIO_MCP_URL.
            youtube_url = os.getenv("YOUTUBE_MCP_URL", composio_mcp_url)
            tools_config["youtube"] = {
                "url": youtube_url,
                "transport": "streamable_http",
                "headers": {
                    "x-api-key": composio_api_key,
                    "x-composio-app": "youtube"
                }
            }
            # If OAuth is enabled for YouTube, attempt to obtain an access token
            try:
                if os.getenv("YOUTUBE_USE_OAUTH", "false").lower() in ("1", "true", "yes"):
                    try:
                        token = get_youtube_access_token()
                        tools_config["youtube"]["headers"]["Authorization"] = f"Bearer {token}"
                        print("Configured YouTube tool with OAuth Authorization header")
                    except Exception as e:
                        print(f"Warning: failed to obtain YouTube OAuth token: {e}")
            except Exception:
                pass
            if progress_callback:
                progress_callback("Added YouTube via Composio MCP... ✅")

        if use_drive:
            drive_url = os.getenv("DRIVE_MCP_URL", composio_mcp_url)
            tools_config["drive"] = {
                "url": drive_url,
                "transport": "streamable_http",
                "headers": {
                    "x-api-key": composio_api_key,
                    "x-composio-app": "googledrive"
                }
            }
            if progress_callback:
                progress_callback("Added Google Drive via Composio MCP... ✅")

        if use_notion:
            notion_url = os.getenv("NOTION_MCP_URL", composio_mcp_url)
            tools_config["notion"] = {
                "url": notion_url,
                "transport": "streamable_http",
                "headers": {
                    "x-api-key": composio_api_key,
                    "x-composio-app": "notion"
                }
            }
            if progress_callback:
                progress_callback("Added Notion via Composio MCP... ✅")

        if not tools_config:
            raise ValueError("At least one integration (YouTube, Drive, or Notion) must be enabled.")

        if progress_callback:
            progress_callback("Initializing Composio MCP client... ✅")

        # Optionally bypass MCP discovery and call tool endpoints directly.
        use_direct = os.getenv("USE_DIRECT_TOOLS", "false").lower() in ("1", "true", "yes")

        # Helper: lightweight direct HTTP tool for calling streamable endpoints
        class DirectHTTPTool:
            def __init__(self, name: str, url: str, headers: Dict[str, str] = None, method: str = "POST"):
                self.name = name
                self.description = f"Direct HTTP tool proxy to {url}"
                self.url = url
                self.headers = headers or {}
                self.method = method.upper()
                # Reuse a single AsyncClient per tool instance to avoid socket exhaustion on Windows
                limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
                self._client = httpx.AsyncClient(timeout=60.0, limits=limits)
                # Register this instance for clean shutdown
                try:
                    DIRECT_HTTP_TOOL_INSTANCES.append(self)
                except Exception:
                    pass

            async def ainvoke(self, params: Any = None):
                # Send params as JSON for POST, or as query params for GET
                try:
                    # Merge in any additional headers provided via env var for this tool
                    env_hdrs_raw = os.getenv(f"{self.name.upper()}_TOOL_HEADERS")
                    extra_hdrs = {}
                    if env_hdrs_raw:
                        try:
                            extra_hdrs = json.loads(env_hdrs_raw)
                        except Exception:
                            extra_hdrs = {}

                    # Copy and update headers
                    req_headers = dict(self.headers or {})
                    req_headers.update(extra_hdrs)

                    # Determine the header name to use for connected account id
                    per_tool_header = os.getenv(f"{self.name.upper()}_CONNECTED_ACCOUNT_HEADER")
                    global_header = os.getenv("CONNECTED_ACCOUNT_HEADER_NAME")
                    connected_header_name = per_tool_header or global_header or "x-composio-connected-account"

                    # If params include a connected_account_id, forward it as the configured header
                    if isinstance(params, dict):
                        conn_id = params.get("connected_account_id") or params.get("connectedAccountId")
                        if conn_id:
                            req_headers.setdefault(connected_header_name, str(conn_id))

                        # Acquire the global semaphore to limit concurrent requests
                        acquired = False
                        try:
                            await _DIRECT_SEMAPHORE.acquire()
                            acquired = True
                        except Exception:
                            acquired = False

                        # Use the shared client for this tool
                        client = self._client
                        # Optional verbose debug of request (toggle with DEBUG_DIRECT_TOOLS env var)
                        debug_enabled = os.getenv("DEBUG_DIRECT_TOOLS", "false").lower() in ("1", "true", "yes")
                        if debug_enabled:
                            safe_params = params
                            try:
                                # Avoid printing large blobs
                                if isinstance(params, (dict, list)):
                                    safe_params = json.dumps(params)[:1000]
                                else:
                                    safe_params = str(params)[:1000]
                            except Exception:
                                safe_params = str(type(params))
                            print(f"DirectHTTPTool('{self.name}') REQUEST -> {self.method} {self.url} | headers={list(req_headers.keys())} | payload={safe_params}")

                        # Perform the request and follow redirects manually up to a limit.
                        max_redirects = 5
                        redirect_count = 0
                        target_url = self.url
                        method = self.method
                        resp = None

                        while True:
                            if method == "GET":
                                resp = await client.get(target_url, params=params or {}, headers=req_headers, follow_redirects=False)
                            else:
                                # POST/other methods
                                if isinstance(params, dict):
                                    resp = await client.post(target_url, json=params, headers=req_headers, follow_redirects=False)
                                else:
                                    resp = await client.post(target_url, content=str(params or ""), headers=req_headers, follow_redirects=False)

                            # If it's not a redirect, break and handle response
                            if not (300 <= resp.status_code < 400):
                                break

                            # It's a redirect. Try to obtain Location header
                            loc = resp.headers.get("location") or resp.headers.get("Location")
                            print(f"DirectHTTPTool('{self.name}') -> {resp.status_code} | headers={list(req_headers.keys())}")
                            if loc:
                                # Resolve relative redirects
                                target_url = urllib.parse.urljoin(target_url, loc)
                                redirect_count += 1
                                print(f"DirectHTTPTool('{self.name}') following redirect to: {target_url} (hop {redirect_count})")
                                # Adjust method for 303 (use GET) per spec; 307/308 keep method
                                if resp.status_code == 303:
                                    method = "GET"
                                if redirect_count > max_redirects:
                                    raise RuntimeError(f"Too many redirects for {self.name}")
                                # loop to follow redirect
                                continue
                            else:
                                # No Location header — treat as failure and break
                                break

                        # Build structured result for the wrapper
                        result = {
                            "successful": 200 <= resp.status_code < 300,
                            "status_code": resp.status_code,
                            "headers": dict(resp.headers),
                            "data": None,
                            "raw": None
                        }

                        ctype = resp.headers.get("content-type", "")
                        try:
                            if "application/json" in ctype.lower():
                                result["data"] = resp.json()
                                result["raw"] = None
                            else:
                                text = resp.text
                                result["raw"] = text
                                # small attempt to parse JSON-like text
                                try:
                                    result["data"] = json.loads(text)
                                except Exception:
                                    result["data"] = None
                        except Exception:
                            result["raw"] = resp.text if hasattr(resp, "text") else str(resp.content)

                        # Debug print for visibility in logs
                        print(f"DirectHTTPTool('{self.name}') -> {resp.status_code} | headers={list(req_headers.keys())}")
                        if debug_enabled:
                            # Print a truncated response body for debugging
                            try:
                                text_preview = resp.text[:2000]
                            except Exception:
                                text_preview = str(resp.content)[:2000]
                            print(f"DirectHTTPTool('{self.name}') RESPONSE PREVIEW: {text_preview}")
                        if not result["successful"]:
                            print(f"DirectHTTPTool('{self.name}') failed: {result['raw'][:300] if result['raw'] else result['data']}")
                        # Release the semaphore if it was acquired
                        if 'acquired' in locals() and acquired:
                            try:
                                _DIRECT_SEMAPHORE.release()
                            except Exception:
                                pass

                        return result
                except Exception as e:
                    raise RuntimeError(f"Direct tool '{self.name}' request failed: {e}") from e
            async def aclose(self):
                try:
                    await self._client.aclose()
                except Exception:
                    pass

        original_tools = []

        if use_direct:
            # Build direct tools from the tools_config map
            for key, cfg_item in tools_config.items():
                url = cfg_item.get("url")
                headers = cfg_item.get("headers", {})
                # Allow a method override via env var if needed
                method = os.getenv(f"{key.upper()}_TOOL_METHOD", "POST")
                original_tools.append(DirectHTTPTool(name=key, url=url, headers=headers, method=method))
        else:
            # Basic network/DNS check for the Composio MCP host to fail fast with a helpful message
            try:
                parsed = urllib.parse.urlparse(composio_mcp_url)
                host = parsed.netloc or parsed.path
                # Remove potential port
                host = host.split(':')[0]
                socket.getaddrinfo(host, None)
            except Exception as dns_exc:
                raise ConnectionError(
                    f"Unable to resolve MCP host '{composio_mcp_url}'. "
                    f"DNS/network check failed with: {dns_exc}.\n"
                    "Please verify your internet connection, DNS settings, and that the MCP URL is correct. "
                    "If you're behind a proxy, ensure environment variables (HTTP_PROXY/HTTPS_PROXY) are set.") from dns_exc

            mcp_client = MultiServerMCPClient(tools_config)
            
            if progress_callback:
                progress_callback("Getting and processing tools... ✅")

            # Get original tools - wrap to provide clearer diagnostics on connection failures
            try:
                original_tools = await mcp_client.get_tools()
            except Exception as conn_err:
                # Common failure is DNS/connection (getaddrinfo/httpx.ConnectError)
                msg = str(conn_err)
                if 'getaddrinfo' in msg or 'gaierror' in msg:
                    raise ConnectionError(
                        "Network error while contacting the Composio MCP server: DNS lookup failed. "
                        "Check the MCP URL, your network, and proxy settings.") from conn_err
                else:
                    # Re-raise with context for easier debugging
                    raise ConnectionError(
                        f"Failed to contact Composio MCP server at {composio_mcp_url}: {conn_err}") from conn_err
        
        print(f"Found {len(original_tools)} original tools")
        for i, tool in enumerate(original_tools):
            print(f"  {i}: {tool.name}")
        
        # Create Composio-aware wrapped tools
        smart_tools = []
        for i, tool in enumerate(original_tools):
            try:
                # Create Composio-aware wrapper with connected account ID
                smart_tool = ComposioAwareTool(
                    tool, 
                    connected_account_id=connected_account_id
                )
                smart_tools.append(smart_tool)
                print(f"✅ Wrapped tool: {tool.name}")
                
            except Exception as e:
                print(f"❌ Failed to wrap tool {tool.name}: {e}")
                continue
        
        if not smart_tools:
            raise Exception("No tools could be successfully wrapped")
        
        if progress_callback:
            progress_callback(f"Successfully processed {len(smart_tools)} tools... ✅")
        
        # ✅ HIGHLIGHT: Use free initialize_model
        model = initialize_model(model_name)
        
        if progress_callback:
            progress_callback("Creating agent with smart tools... ✅")
        
        # Create agent
        agent = create_react_agent(model, smart_tools)
        
        if progress_callback:
            progress_callback("Agent setup complete! ✅")
        
        return agent
        
    except Exception as e:
        print(f"Error in setup_agent_with_tools: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def run_agent_sync(
    # google_api_key: str,
    use_youtube: bool = True,
    use_drive: bool = False,
    use_notion: bool = False,
    user_goal: str = "",
    connected_account_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    model_name: str = "gemini-2.5-flash"
) -> str: # 🌟
    """
    Synchronous wrapper for running the agent with Composio MCP.
    """
    async def _run():
        try:
            agent = await setup_agent_with_tools(
                # google_api_key=google_api_key,
                use_youtube=use_youtube,
                use_drive=use_drive,
                use_notion=use_notion,
                connected_account_id=connected_account_id,
                progress_callback=progress_callback,
                model_name=model_name
            )
            
            default_days = int(os.getenv("DESIRED_DAYS", "10"))
            desired_days = infer_requested_day_count(user_goal, default_days)

            enhanced_prompt = f"""User Goal: {user_goal}

{user_goal_prompt}

OUTPUT REQUIREMENTS (READ CAREFULLY):
1. Produce EXACTLY {desired_days} day(s) in this response. If the user asked for another number, honor it. Never ask the user to continue.
2. Every day must include: a heading with the day number and topic, a short focus/summary sentence, **at least one YouTube video** with a canonical URL, **at least one non-video resource** (course, article, documentation, newsletter, etc.), a practice/reflection task, and 2-3 concise learning objectives.
3. For each resource, format as `- [Title](URL) — Type: <Video/Course/Article/...> — Reason it helps`. Ensure at least one resource per day is not from YouTube and annotate each video with its publication year or age (e.g., "Uploaded 2024").
4. Only include real, reachable links. If you cannot find a link, replace the resource with another option rather than emitting placeholders.
5. Use clean markdown with `## Day X - Topic` headings. After listing all days, add a short "Summary & Playlists" section that mentions any document/page or playlist URLs you actually created (omit if none).
6. Use connected_account_id as needed for tool execution and avoid exposing API responses or debug info.
7. Guarantee that every day references a distinct, currently available YouTube video younger than ~24 months; if you detect a duplicate or stale link, re-run your search and swap it for a newer upload before finalizing the answer.
8. Never mention a video without its canonical `https://www.youtube.com/watch?v=VIDEO_ID` (or `https://youtu.be/VIDEO_ID`) link; if you cannot provide the exact link, choose a different recent video.

Respond in English and include all {desired_days} day(s) now.
"""
            
            if progress_callback:
                progress_callback("Generating your learning path...")

            # Collect playlist events for UI reporting
            playlist_events: list[dict] = []
            
            # Run the agent
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=enhanced_prompt)]},
                config=cfg
            )

            # Helper to extract the final text from an agent response
            def _extract_text_from_result(res) -> Optional[str]:
                if isinstance(res, dict) and "messages" in res:
                    final_message = res["messages"][-1]
                    if hasattr(final_message, "content"):
                        return final_message.content
                    # sometimes it's a plain dict message
                    if isinstance(final_message, dict) and "content" in final_message:
                        return final_message.get("content")
                elif isinstance(res, str):
                    return res
                return None

            first_text = _extract_text_from_result(result)

            # Iterative continuation: if model returned fewer days than desired, ask it to continue
            max_iter = int(os.getenv("CONTINUATION_MAX_ITER", "4"))
            combined_text = first_text or ""
            try:
                import re as _re
                day_nums = [int(m.group(1)) for m in _re.finditer(r"##\s*Day\s*(\d+)", combined_text)]
                max_day = max(day_nums) if day_nums else 0
            except Exception:
                max_day = 0

            iter_count = 0
            while max_day < desired_days and iter_count < max_iter:
                iter_count += 1
                next_start = max_day + 1
                cont_prompt = f"Continue the learning path in the exact same format starting from Day {next_start} through Day {desired_days}. Do NOT repeat earlier days. Provide only the continuation in the required markdown format."
                if progress_callback:
                    progress_callback(f"Requesting continuation from model: Day {next_start} to {desired_days} (attempt {iter_count})")
                cont_res = await agent.ainvoke({"messages": [HumanMessage(content=cont_prompt)]}, config=cfg)
                cont_text = _extract_text_from_result(cont_res)
                if not cont_text:
                    break
                # Append with a newline separator
                combined_text = combined_text.rstrip() + "\n\n" + cont_text.lstrip()
                # Recompute max_day
                try:
                    day_nums = [int(m.group(1)) for m in _re.finditer(r"##\s*Day\s*(\d+)", combined_text)]
                    max_day = max(day_nums) if day_nums else max_day
                except Exception:
                    break

            final_text = combined_text

            
            if progress_callback:
                progress_callback("Learning path generation complete!")
            
            # Extract the final content from the agent's response
            if isinstance(result, dict) and "messages" in result:
                # The final answer is typically in the content of the last message
                final_message = result["messages"][-1]
                if hasattr(final_message, "content"):
                    final_text = final_message.content

                    # Post-process playlists and video links to keep generated playlists populated.
                    try:
                        playlist_refs = re.findall(r"playlist\?list=([A-Za-z0-9_\-]+)", final_text)
                        if "PLAYLIST_ID" in final_text or "[YouTube]" in final_text:
                            playlist_refs.append("PLAYLIST_ID")
                        for m in re.finditer(r"<([^>]{3,200})>", final_text):
                            token = m.group(1).strip()
                            if token.upper().startswith("INSERT") and "PLAYLIST" in token.upper():
                                playlist_refs.append(token)
                        if "INSERT_YOUTUBE_PLAYLIST_LINK" in final_text:
                            playlist_refs.append("INSERT_YOUTUBE_PLAYLIST_LINK")

                        video_title_candidates = collect_video_title_candidates(final_text)
                        vids = extract_video_ids_from_text(final_text, title_hints=video_title_candidates)
                        vids = list(dict.fromkeys(vids))

                        placeholder_tokens = {"PLAYLIST_ID", "[YouTube]", "INSERT_YOUTUBE_PLAYLIST_LINK"}

                        removed_unavailable: list[str] = []
                        if vids:
                            vids, removed_unavailable = filter_available_videos(vids)
                            if removed_unavailable:
                                playlist_events.append({"action": "filtered_unavailable", "removed": removed_unavailable})
                                try:
                                    for rem in removed_unavailable:
                                        print(f"[Playlist] Filtered unavailable video id: {rem}")
                                except Exception:
                                    pass
                        if (not vids or removed_unavailable) and video_title_candidates:
                            exclude = set(vids or []) | set(removed_unavailable)
                            replacements: list[str] = []
                            for title in video_title_candidates:
                                vid = search_youtube_for_title(title, exclude_ids=exclude)
                                if vid and vid not in exclude:
                                    replacements.append(vid)
                                    exclude.add(vid)
                            if replacements:
                                vids.extend(replacements)
                                vids = list(dict.fromkeys(vids))
                                vids, removed_retry = filter_available_videos(vids)
                                if removed_retry:
                                    playlist_events.append({"action": "filtered_unavailable_retry", "removed": removed_retry})
                                    for rem in removed_retry:
                                        if rem not in removed_unavailable:
                                            removed_unavailable.append(rem)

                        if not vids:
                            playlist_events.append({"action": "no_valid_videos"})

                        def _sanitize_playlist_token(raw_token: str) -> Optional[str]:
                            if not raw_token:
                                return None
                            token = raw_token.strip()
                            if token in placeholder_tokens:
                                return None
                            token = token.replace("https://www.youtube.com/playlist?list=", "")
                            token = token.replace("http://www.youtube.com/playlist?list=", "")
                            token = token.replace("www.youtube.com/playlist?list=", "")
                            token = token.replace("youtube.com/playlist?list=", "")
                            token = re.sub(r"[^A-Za-z0-9_-]", "", token)
                            if re.match(r"^[A-Za-z0-9_-]{10,}$", token):
                                return token
                            if "_" in token:
                                candidate = token.split("_", 1)[0]
                                if re.match(r"^[A-Za-z0-9_-]{10,}$", candidate):
                                    return candidate
                            return None

                        unique_refs = list(dict.fromkeys(playlist_refs))
                        processed_existing: set[str] = set()
                        placeholder_creations: dict[str, str] = {}

                        if vids and unique_refs:
                            for raw_ref in unique_refs:
                                try:
                                    clean_id = _sanitize_playlist_token(raw_ref)
                                    if clean_id:
                                        if clean_id in processed_existing:
                                            continue
                                        processed_existing.add(clean_id)
                                        try:
                                            existing_vids = get_playlist_item_video_ids(clean_id)
                                        except Exception:
                                            existing_vids = []
                                        existing_set = set(existing_vids)
                                        to_add = [v for v in vids if v not in existing_set]
                                        if not to_add:
                                            continue
                                        added_info = add_videos_to_playlist(clean_id, to_add)
                                        evt = {
                                            "action": "added_missing",
                                            "playlist_id": clean_id,
                                            "added": added_info.get("added", []),
                                            "failed": added_info.get("failed", []),
                                            "already_present": [v for v in vids if v in existing_set],
                                        }
                                        playlist_events.append(evt)
                                        try:
                                            for added_vid in evt.get("added", []):
                                                print(f"[Playlist][{clean_id}] Added video: https://www.youtube.com/watch?v={added_vid}")
                                            for failed_vid in evt.get("failed", []):
                                                print(f"[Playlist][{clean_id}] Failed to add video: {failed_vid}")
                                        except Exception:
                                            pass
                                        if progress_callback:
                                            progress_callback(f"Playlist {clean_id}: added {len(evt['added'])} videos, failed {len(evt['failed'])}")
                                    else:
                                        if raw_ref in placeholder_creations:
                                            continue
                                        try:
                                            new_pid = create_youtube_playlist_and_add_videos(vids, title=f"Learning Path for {user_goal}")
                                            placeholder_creations[raw_ref] = new_pid
                                            evt = {"action": "created_and_added", "playlist_id": new_pid, "added": vids}
                                            playlist_events.append(evt)
                                            try:
                                                for v in vids:
                                                    print(f"[Playlist][{new_pid}] Added video: https://www.youtube.com/watch?v={v}")
                                                print(f"[Playlist][{new_pid}] Created playlist https://www.youtube.com/playlist?list={new_pid}")
                                            except Exception:
                                                pass
                                            if progress_callback:
                                                progress_callback(f"Created playlist {new_pid} and added {len(vids)} videos")
                                            playlist_url = f"https://www.youtube.com/playlist?list={new_pid}"
                                            if raw_ref == "PLAYLIST_ID":
                                                final_text = final_text.replace("PLAYLIST_ID", new_pid)
                                            elif raw_ref == "[YouTube]":
                                                final_text = final_text.replace("[YouTube]", playlist_url)
                                            else:
                                                final_text = final_text.replace(raw_ref, playlist_url)
                                                final_text = final_text.replace(f"<{raw_ref}>", playlist_url)
                                            final_text = final_text.replace("INSERT_YOUTUBE_PLAYLIST_LINK", playlist_url)
                                        except Exception as create_exc:
                                            playlist_events.append({"action": "create_failed", "raw": raw_ref, "error": str(create_exc)})
                                            try:
                                                print(f"[Playlist] Failed to create playlist for reference '{raw_ref}': {create_exc}")
                                            except Exception:
                                                pass
                                            if progress_callback:
                                                progress_callback(f"Failed to create playlist for reference '{raw_ref}': {create_exc}")
                                except Exception as proc_exc:
                                    playlist_events.append({"action": "processing_error", "raw": raw_ref, "error": str(proc_exc)})
                                    try:
                                        print(f"[Playlist] Error processing playlist reference '{raw_ref}': {proc_exc}")
                                    except Exception:
                                        pass
                                    if progress_callback:
                                        progress_callback(f"Error processing playlist reference '{raw_ref}': {proc_exc}")
                        if vids and not any(evt.get("playlist_id") for evt in playlist_events):
                            try:
                                fallback_pid = create_youtube_playlist_and_add_videos(vids, title=f"Learning Path for {user_goal}")
                                playlist_events.append({"action": "created_without_reference", "playlist_id": fallback_pid, "added": vids})
                                try:
                                    print(f"[Playlist][{fallback_pid}] Created playlist without placeholder reference")
                                except Exception:
                                    pass
                                if progress_callback:
                                    progress_callback(f"Created playlist {fallback_pid} with {len(vids)} videos")
                            except Exception as fallback_exc:
                                playlist_events.append({"action": "create_failed", "raw": "auto", "error": str(fallback_exc)})
                                try:
                                    print(f"[Playlist] Failed to auto-create playlist: {fallback_exc}")
                                except Exception:
                                    pass
                                if progress_callback:
                                    progress_callback(f"Failed to auto-create playlist: {fallback_exc}")
                    except Exception:
                        pass

                    # Append canonical playlist links at the bottom of the response for visibility
                    try:
                        playlist_links = []
                        for evt in playlist_events:
                            pid = evt.get("playlist_id")
                            if pid:
                                url = f"https://www.youtube.com/playlist?list={pid}"
                                playlist_links.append(url)
                        if playlist_links:
                            final_text = final_text.rstrip() + "\n\n" + "\n".join([f"Playlist: {u}" for u in playlist_links])
                    except Exception:
                        pass

                    final_text = remove_placeholder_lines(final_text)

                    # Return both the final text and any playlist events that occurred
                    return {"text": final_text, "playlists": playlist_events}
                else:
                    return str(result) # Fallback if content not found
            else:
                return str(result) # Fallback if structure is unexpected
            
        except Exception as e:
            print(f"Error in _run: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # Run in new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()