"""
actions.py — Action layer for the Learning Path Generator.

This module performs all writes to external services (YouTube, Google Drive,
Notion) using the authenticated user's Composio connected account.

CRITICAL RULES:
- Every function takes user_id as the first argument.
- Every function resolves connectedAccountId from the DB via composio_client.
- Developer credentials are NEVER used here.
- No global tokens, no shared accounts.

Error model:
- Functions return typed result dicts rather than raising on partial failure.
- Each result always has a 'success' bool.
- Non-fatal errors are logged and returned in the result for UI display.
"""
from __future__ import annotations

import re
from typing import Optional

import composio_client as cc

# ---------------------------------------------------------------------------
# YouTube actions
# ---------------------------------------------------------------------------

def create_youtube_playlist(
    user_id: int,
    video_ids: list[str],
    title: str,
    description: str = "Created by MCP Learning Path Generator",
) -> dict:
    """Create a public YouTube playlist in the user's account and add videos.

    Args:
        user_id:   App-level user ID.
        video_ids: Validated, available YouTube video IDs to add.
        title:     Playlist title (e.g. "Learn Python in 10 Days").
        description: Optional description.

    Returns:
        {
            'success': bool,
            'playlist_id': str | None,
            'playlist_url': str | None,
            'added': list[str],       # video IDs successfully added
            'failed': list[str],      # video IDs that failed to add
            'error': str | None,      # top-level error message if success=False
        }
    """
    result: dict = {
        "success": False,
        "playlist_id": None,
        "playlist_url": None,
        "added": [],
        "failed": [],
        "error": None,
    }

    # Step 1: Create playlist
    try:
        resp = cc.execute_tool(
            user_id=user_id,
            provider="youtube",
            action="YOUTUBE_CREATE_PLAYLIST",
            params={
                "title": title,
                "description": description,
                "privacyStatus": "public",
            },
        )
        playlist_id, err = _parse_composio_response(resp, "youtube")
        if err:
            result["error"] = err
            return result

        result["playlist_id"] = playlist_id
        result["playlist_url"] = f"https://www.youtube.com/playlist?list={playlist_id}"

    except RuntimeError as e:
        result["error"] = str(e)
        return result

    # Step 2: Add videos one by one (continue on individual failures)
    for vid in video_ids:
        try:
            cc.execute_tool(
                user_id=user_id,
                provider="youtube",
                action="YOUTUBE_ADD_VIDEO_TO_PLAYLIST",
                params={
                    "playlistId": playlist_id,
                    "videoId": vid,
                },
            )
            result["added"].append(vid)
        except Exception as e:
            result["failed"].append(vid)
            print(f"[actions] Failed to add video {vid} to playlist {playlist_id}: {e}")

    result["success"] = True
    return result


def _extract_id_from_response(resp: dict) -> Optional[str]:
    """Try to find a playlist/resource ID anywhere in a Composio response dict."""
    text = str(resp)
    # Look for PLxxxxxxxx pattern
    match = re.search(r"(PL[A-Za-z0-9_\-]{10,})", text)
    if match:
        return match.group(1)
    return None


def _parse_composio_response(resp: dict, provider: str) -> tuple[Optional[str], Optional[str]]:
    """Parse a Composio execution response and return (resource_id, error_message)."""
    if not isinstance(resp, dict):
        return None, f"Invalid response format: {resp}"

    # Check top-level success if present
    is_successful = resp.get("successful")
    if is_successful is None:
        is_successful = resp.get("success")
    if is_successful is False:
        return None, resp.get("error") or resp.get("message") or f"Action failed: {resp}"

    # Extract data block
    data = None
    for key in ["response", "data"]:
        val = resp.get(key)
        if isinstance(val, dict):
            data = val
            break

    if data is None:
        data = resp

    # Check for error fields inside the data block
    error_msg = None
    if isinstance(data, dict):
        if "error" in data:
            error_msg = str(data["error"])
        elif "message" in data:
            error_msg = str(data["message"])
        elif "detail" in data:
            error_msg = str(data["detail"])

    # Resolve resource ID based on provider
    resource_id = None
    if isinstance(data, dict):
        resource_id = data.get("id")
        if not resource_id and provider == "youtube":
            snippet = data.get("snippet", {})
            if isinstance(snippet, dict):
                resource_id = snippet.get("id")

    # Fallback to regex extraction if not found directly
    if not resource_id:
        if provider == "youtube":
            resource_id = _extract_id_from_response(resp)
        elif provider == "googledrive":
            resource_id = _extract_drive_id(resp)
        elif provider == "notion":
            resource_id = _extract_notion_id(resp)

    # If we have an error message and no resource_id was resolved, return the error
    if error_msg and not resource_id:
        return None, error_msg

    # If no ID was resolved and we have no explicit error, return failure
    if not resource_id:
        if error_msg:
            return None, error_msg
        return None, f"No resource ID found in response: {str(resp)[:300]}"

    return resource_id, None


# ---------------------------------------------------------------------------
# Google Drive actions
# ---------------------------------------------------------------------------

def create_google_doc(
    user_id: int,
    title: str,
    markdown_content: str,
) -> dict:
    """Create a Google Doc in the user's Drive with the given markdown content.

    Args:
        user_id:          App-level user ID.
        title:            Document title.
        markdown_content: Learning path in markdown format.

    Returns:
        {
            'success': bool,
            'doc_id': str | None,
            'doc_url': str | None,
            'error': str | None,
        }
    """
    result: dict = {
        "success": False,
        "doc_id": None,
        "doc_url": None,
        "error": None,
    }

    try:
        resp = cc.execute_tool(
            user_id=user_id,
            provider="googledrive",
            action="GOOGLEDRIVE_CREATE_FILE_FROM_TEXT",
            params={
                "file_name": title,
                "text_content": markdown_content,
            },
        )
        doc_id, err = _parse_composio_response(resp, "googledrive")
        if err:
            result["error"] = err
            return result

        result["doc_id"] = doc_id
        result["doc_url"] = f"https://docs.google.com/document/d/{doc_id}/edit"
        result["success"] = True

    except RuntimeError as e:
        result["error"] = str(e)

    return result


def _extract_drive_id(resp: dict) -> Optional[str]:
    """Try to find a Drive file ID in a Composio response."""
    text = str(resp)
    # Drive IDs are typically long alphanumeric strings
    match = re.search(r'"id"\s*:\s*"([A-Za-z0-9_\-]{20,})"', text)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Notion actions
# ---------------------------------------------------------------------------

def create_notion_page(
    user_id: int,
    title: str,
    markdown_content: str,
    parent_page_id: Optional[str] = None,
) -> dict:
    """Create a Notion page in the user's workspace.

    Args:
        user_id:          App-level user ID.
        title:            Page title.
        markdown_content: Learning path in markdown format.
        parent_page_id:   Optional Notion page/database ID to create inside.
                          If None, creates at workspace root.

    Returns:
        {
            'success': bool,
            'page_id': str | None,
            'page_url': str | None,
            'error': str | None,
        }
    """
    result: dict = {
        "success": False,
        "page_id": None,
        "page_url": None,
        "error": None,
    }

    create_params: dict = {
        "title": title,
    }
    if parent_page_id:
        create_params["parent_id"] = parent_page_id

    try:
        resp = cc.execute_tool(
            user_id=user_id,
            provider="notion",
            action="NOTION_CREATE_NOTION_PAGE",
            params=create_params,
        )
        page_id, err = _parse_composio_response(resp, "notion")
        if err:
            result["error"] = err
            return result

        result["page_id"] = page_id
        clean_id = page_id.replace("-", "")
        result["page_url"] = f"https://notion.so/{clean_id}"

        # Step 2: Add page content using natural language text execution
        try:
            cc.execute_tool(
                user_id=user_id,
                provider="notion",
                action="NOTION_ADD_MULTIPLE_PAGE_CONTENT",
                text_instruction=f"Add the following markdown text as blocks to the page with ID {page_id}:\n\n{markdown_content}"
            )
            result["success"] = True
        except Exception as e:
            result["success"] = True
            result["error"] = f"Page created, but failed to append content: {e}"

    except RuntimeError as e:
        result["error"] = str(e)

    return result


def _extract_notion_id(resp: dict) -> Optional[str]:
    """Try to find a Notion page ID (UUID format) in a Composio response."""
    text = str(resp)
    match = re.search(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", text, re.IGNORECASE
    )
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Orchestrated post-generation actions
# ---------------------------------------------------------------------------

def run_post_generation_actions(
    user_id: int,
    goal: str,
    markdown: str,
    video_ids: list[str],
    connection_status: dict[str, bool],
) -> dict:
    """Run all connected post-generation actions for a user.

    This is the single entry point called by the app after learning path
    generation is complete. It respects which services the user has connected.

    Args:
        user_id:           App-level user ID.
        goal:              The user's original learning goal (used in titles).
        markdown:          The full markdown learning path.
        video_ids:         Validated YouTube video IDs to add to playlist.
        connection_status: Dict from composio_client.get_connection_status().

    Returns:
        {
            'playlist': dict | None,
            'google_doc': dict | None,
            'notion_page': dict | None,
        }
    """
    results: dict = {
        "playlist": None,
        "google_doc": None,
        "notion_page": None,
    }
    title = f"{goal} — Learning Path"

    if connection_status.get("youtube") and video_ids:
        results["playlist"] = create_youtube_playlist(
            user_id=user_id,
            video_ids=video_ids,
            title=title,
        )

    if connection_status.get("googledrive"):
        results["google_doc"] = create_google_doc(
            user_id=user_id,
            title=title,
            markdown_content=markdown,
        )

    if connection_status.get("notion"):
        results["notion_page"] = create_notion_page(
            user_id=user_id,
            title=title,
            markdown_content=markdown,
        )

    return results
