"""
composio_client.py — Composio API wrapper for the Learning Path Generator.

This module is the single point of contact with Composio.
- No Composio internal URLs or credentials are exposed to the UI layer.
- Every action is executed using the user's connectedAccountId resolved from the DB.
- Developer credentials are never used for user-facing actions.

Supported providers: youtube, googledrive, notion
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

import db

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COMPOSIO_API_BASE = "https://backend.composio.dev/api/v1"
_COMPOSIO_MCP_BASE = "https://mcp.composio.dev"

# Provider → Composio app slug mapping
_PROVIDER_APP_SLUG: dict[str, str] = {
    "youtube": "youtube",
    "googledrive": "googledrive",
    "notion": "notion",
}

# Provider → Composio integration ID env var
_PROVIDER_INTEGRATION_ENV: dict[str, str] = {
    "youtube": "COMPOSIO_YOUTUBE_INTEGRATION_ID",
    "googledrive": "COMPOSIO_DRIVE_INTEGRATION_ID",
    "notion": "COMPOSIO_NOTION_INTEGRATION_ID",
}


def _api_key() -> str:
    key = os.getenv("COMPOSIO_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("COMPOSIO_API_KEY is not set in the environment.")
    return key


def _integration_id(provider: str) -> Optional[str]:
    env_var = _PROVIDER_INTEGRATION_ENV.get(provider.lower())
    if not env_var:
        return None
    return os.getenv(env_var, "").strip() or None


# ---------------------------------------------------------------------------
# OAuth flow helpers
# ---------------------------------------------------------------------------

def get_oauth_url(user_id: int, provider: str, redirect_url: Optional[str] = None) -> str:
    """Return the Composio OAuth authorization URL for the given provider.

    The user should be redirected to this URL to begin the OAuth flow.
    After authorization, Composio calls the redirect_url with the connected
    account details; call handle_oauth_callback() with the response.

    Args:
        user_id:      App-level user ID (used as entity_id in Composio).
        provider:     One of 'youtube', 'googledrive', 'notion'.
        redirect_url: Where Composio should redirect after auth. If None,
                      Composio's default redirect is used.

    Returns:
        The URL string the user should open in their browser.
    """
    provider = provider.lower()
    app_slug = _PROVIDER_APP_SLUG.get(provider)
    if not app_slug:
        raise ValueError(f"Unsupported provider: {provider!r}")

    integration_id = _integration_id(provider)
    entity_id = f"user_{user_id}"

    payload: dict[str, Any] = {
        "appName": app_slug,
        "entityId": entity_id,
    }
    if integration_id:
        payload["integrationId"] = integration_id
    if redirect_url:
        payload["redirectUri"] = redirect_url

    try:
        resp = httpx.post(
            f"{_COMPOSIO_API_BASE}/connectedAccounts",
            headers={
                "x-api-key": _api_key(),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        oauth_url = data.get("redirectUrl") or data.get("authorizationUrl") or data.get("url")
        if not oauth_url:
            raise RuntimeError(f"Composio returned no OAuth URL. Response: {data}")
        return oauth_url
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Failed to initiate OAuth for {provider}: HTTP {e.response.status_code} — {e.response.text}"
        ) from e


def handle_oauth_callback(user_id: int, provider: str, connected_account_id: str) -> None:
    """Store the connectedAccountId returned by Composio after OAuth authorization.

    Call this after the OAuth redirect returns to your app with the
    connected_account_id (either via query param or Composio webhook).

    Args:
        user_id:              App-level user ID.
        provider:             One of 'youtube', 'googledrive', 'notion'.
        connected_account_id: The ID returned by Composio.
    """
    provider = provider.lower()
    if provider not in _PROVIDER_APP_SLUG:
        raise ValueError(f"Unsupported provider: {provider!r}")
    db.save_oauth_connection(user_id, provider, connected_account_id)


def get_connection_status(user_id: int) -> dict[str, bool]:
    """Return connection status dict for all providers.

    Returns: {'youtube': bool, 'googledrive': bool, 'notion': bool}
    """
    return db.get_connection_status(user_id)


def disconnect(user_id: int, provider: str) -> None:
    """Remove the stored connection for a user+provider pair."""
    db.delete_connection(user_id, provider.lower())


# ---------------------------------------------------------------------------
# MCP tool execution
# ---------------------------------------------------------------------------

def _resolve_connected_account_id(user_id: int, provider: str) -> str:
    """Look up the user's connectedAccountId for a provider.

    Raises RuntimeError if no connection is found.
    """
    cid = db.get_connection(user_id, provider.lower())
    if not cid:
        raise RuntimeError(
            f"User {user_id} has no connected {provider!r} account. "
            "Please connect the integration first."
        )
    return cid


def execute_tool(
    user_id: int,
    provider: str,
    action: str,
    params: dict[str, Any],
) -> Any:
    """Execute a Composio MCP tool action on behalf of the user.

    The user's connectedAccountId is resolved from the database — developer
    credentials are never substituted.

    Args:
        user_id:  App-level user ID.
        provider: Tool provider ('youtube', 'googledrive', 'notion').
        action:   Composio action name (e.g. 'YOUTUBE_CREATE_PLAYLIST').
        params:   Action-specific parameters dict.

    Returns:
        The parsed JSON response from Composio.

    Raises:
        RuntimeError: If the user has no connection or the API call fails.
    """
    connected_account_id = _resolve_connected_account_id(user_id, provider)

    try:
        resp = httpx.post(
            f"{_COMPOSIO_API_BASE}/actions/{action}/execute",
            headers={
                "x-api-key": _api_key(),
                "Content-Type": "application/json",
            },
            json={
                "connectedAccountId": connected_account_id,
                "input": params,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        body = e.response.text[:500]
        if status == 401:
            # Token expired — remove stale connection so user knows to reconnect
            db.delete_connection(user_id, provider.lower())
            raise RuntimeError(
                f"OAuth token for {provider!r} has expired or been revoked. "
                "Please reconnect via the Integrations panel."
            ) from e
        if status == 429:
            raise RuntimeError(
                f"API quota exhausted for {provider!r}. Please wait and try again."
            ) from e
        raise RuntimeError(
            f"Composio action {action!r} failed: HTTP {status} — {body}"
        ) from e
    except httpx.TimeoutException as e:
        raise RuntimeError(
            f"Request to Composio timed out while executing {action!r}."
        ) from e


# ---------------------------------------------------------------------------
# Convenience: verify a connected account is still valid
# ---------------------------------------------------------------------------

def verify_connection(user_id: int, provider: str) -> bool:
    """Return True if the stored connection appears valid (basic API check).

    This is a lightweight check — it verifies the connectedAccountId exists in
    Composio. It does not guarantee the underlying OAuth token hasn't expired.
    """
    try:
        cid = db.get_connection(user_id, provider.lower())
        if not cid:
            return False
        resp = httpx.get(
            f"{_COMPOSIO_API_BASE}/connectedAccounts/{cid}",
            headers={"x-api-key": _api_key()},
            timeout=10.0,
        )
        return resp.status_code == 200
    except Exception:
        return False
