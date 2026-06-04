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
import datetime
from streamlit_cookies_controller import CookieController

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
    
    # Try restoring session from browser cookie
    if get_current_user() is None:
        try:
            controller = CookieController()
            saved_email = controller.get("auth_user_email")
            if saved_email:
                user = db.get_or_create_user(saved_email)
                st.session_state[_SESSION_KEY] = user
        except Exception:
            pass
    
    # Auto-login from query parameter if user session is lost
    params = st.query_params
    if get_current_user() is None and "login_email" in params:
        email = params["login_email"]
        try:
            user = db.get_or_create_user(email)
            st.session_state[_SESSION_KEY] = user
            
            # Save to cookie for 30 days
            try:
                controller = CookieController()
                expires = datetime.datetime.now() + datetime.timedelta(days=30)
                controller.set("auth_user_email", email, expires=expires)
            except Exception:
                pass
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
    if "login_email" in st.query_params:
        del st.query_params["login_email"]
    try:
        controller = CookieController()
        controller.remove("auth_user_email")
    except Exception:
        pass


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
    
    # Robust RFC 5322 compliant regex for email validation
    import re
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not email or not re.match(email_regex, email):
        st.error("Please enter a valid email address (e.g. name@example.com).")
        return
        
    user = db.get_or_create_user(email)
    st.session_state[_SESSION_KEY] = user
    
    # Save to cookie for 30 days
    try:
        controller = CookieController()
        expires = datetime.datetime.now() + datetime.timedelta(days=30)
        controller.set("auth_user_email", email, expires=expires)
    except Exception:
        pass
        
    st.query_params["login_email"] = email
    st.rerun()


def _login_ui() -> None:
    """Render the login form."""
    st.markdown(
        """
        <style>
        .login-card {
            max-width: 480px;
            margin: 6rem auto 1.5rem auto;
            padding: 2.5rem;
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 24px;
            backdrop-filter: blur(20px);
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.45), 0 0 40px rgba(139, 92, 246, 0.1);
            text-align: center;
        }
        .login-card h2 {
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0 0 0.6rem 0;
            background: linear-gradient(135deg, #a78bfa 0%, #22d3ee 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .login-card p {
            color: rgba(248, 250, 252, 0.7);
            margin: 0;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        /* Style text input inside login flow */
        div[data-testid="stTextInput"] input {
            background: rgba(3, 7, 18, 0.45) !important;
            border: 1px solid rgba(148, 163, 184, 0.15) !important;
            border-radius: 99px !important;
            color: #f8fafc !important;
            padding: 0.65rem 1.4rem !important;
            font-size: 0.95rem !important;
            text-align: center !important;
            transition: all 0.25s ease !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: rgba(139, 92, 246, 0.5) !important;
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.2) !important;
        }
        </style>
        <div class="login-card">
            <h2>🧭 Learning Path Generator</h2>
            <p>Enter your email below to start generating personalized learning paths, Google docs, and playlists.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        email = st.text_input(
            "Email address",
            placeholder="you@example.com",
            key="login_email_input",
            label_visibility="collapsed",
        )
        if st.button("Get Started →", use_container_width=True, type="primary", key="login_btn"):
            with st.spinner("Signing you in…"):
                _do_login(email)
        st.caption("<div style='text-align:center; color:#64748b; margin-top:0.4rem;'>No password required during preview. Your email is your unique identity.</div>", unsafe_allow_html=True)

