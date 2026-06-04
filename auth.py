"""
auth.py — Application-level authentication for the Learning Path Generator.

Uses Streamlit session state to maintain login. No passwords in Phase 1 —
users are identified by email only (email = unique identity).

Dev bypass:
    Set DEV_USER_EMAIL in .env to skip the login screen entirely.
    The dev user is auto-created in the DB and auto-logged-in on every run.
"""
from __future__ import annotations

import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

import db

load_dotenv()

_SESSION_KEY = "auth_user"
_DEV_EMAIL_VAR = "DEV_USER_EMAIL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_current_user() -> Optional[dict]:
    """Return the current logged-in user dict or None."""
    return st.session_state.get(_SESSION_KEY)


def require_login() -> dict:
    """Block execution and show login UI if the user is not logged in.

    Call this at the top of any page that requires authentication.
    Returns the user dict when logged in.
    """
    _maybe_dev_autologin()
    
    # Auto-login from query parameter if user session is lost
    params = st.query_params
    if get_current_user() is None and "login_email" in params:
        email = params["login_email"]
        try:
            user = db.get_or_create_user(email)
            st.session_state[_SESSION_KEY] = user
        except Exception:
            pass

    user = get_current_user()
    if user is None:
        _login_ui()
        st.stop()
    return user


def logout() -> None:
    """Clear the current session."""
    st.session_state.pop(_SESSION_KEY, None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _maybe_dev_autologin() -> None:
    """If DEV_USER_EMAIL is set, automatically log in as that user."""
    if get_current_user() is not None:
        return
    dev_email = os.getenv(_DEV_EMAIL_VAR, "").strip()
    if dev_email:
        user = db.get_or_create_user(dev_email, name="Dev User")
        st.session_state[_SESSION_KEY] = user


def _do_login(email: str) -> None:
    """Validate email, create/fetch user, store in session."""
    email = email.strip().lower()
    if not email or "@" not in email:
        st.error("Please enter a valid email address.")
        return
    user = db.get_or_create_user(email)
    st.session_state[_SESSION_KEY] = user
    st.rerun()


def _login_ui() -> None:
    """Render the login form."""
    st.markdown(
        """
        <style>
        .login-wrapper {
            max-width: 420px;
            margin: 8rem auto 0 auto;
            text-align: center;
        }
        .login-wrapper h2 {
            font-size: 1.8rem;
            margin-bottom: 0.4rem;
        }
        .login-wrapper p {
            color: rgba(226,232,240,0.7);
            margin-bottom: 1.8rem;
            font-size: 0.95rem;
        }
        </style>
        <div class="login-wrapper">
            <h2>🧭 Learning Path Generator</h2>
            <p>Sign in to connect your accounts and generate personalised learning paths.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        email = st.text_input(
            "Email address",
            placeholder="you@example.com",
            key="login_email_input",
            label_visibility="collapsed",
        )
        if st.button("Continue →", use_container_width=True, type="primary"):
            _do_login(email)
        st.caption("No password required during preview. Your email is your identity.")
