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
    # Try restoring session from query parameters (specifically for OAuth redirects/callbacks or login reruns)
    if get_current_user() is None:
        try:
            token_from_url = st.query_params.get("session_token")
            if token_from_url:
                user = db.get_user_by_session_token(token_from_url)
                if user:
                    st.session_state[_SESSION_KEY] = user
                    try:
                        controller = CookieController()
                        expires = datetime.datetime.now() + datetime.timedelta(days=5)
                        controller.set("auth_session_token", token_from_url, expires=expires)
                    except Exception:
                        pass
                    
                    # Clear session_token from query parameters to prevent leakage in URL bar, keeping other params
                    new_params = {k: v for k, v in st.query_params.items() if k != "session_token"}
                    st.query_params.clear()
                    for k, v in new_params.items():
                        st.query_params[k] = v
        except Exception:
            pass

    # Try restoring session from browser cookie
    if get_current_user() is None:
        try:
            controller = CookieController()
            saved_token = controller.get("auth_session_token")
            if saved_token:
                user = db.get_user_by_session_token(saved_token)
                if user:
                    st.session_state[_SESSION_KEY] = user
        except Exception:
            pass
    
    # Auto-login from cookie is handled above. 
    # Removed insecure fallback from query_params to prevent session fixation/URL leakage.

    user = get_current_user()
    if user is None:
        # Workaround for Streamlit's asynchronous custom components.
        # On a hard refresh or OAuth redirect, the CookieController takes a fraction
        # of a second to send the cookie from the browser to the Python backend.
        # If we instantly show the login screen, we break the OAuth callback flow.
        if "cookie_waited" not in st.session_state:
            st.session_state["cookie_waited"] = True
            with st.spinner("Restoring session..."):
                pass
            # Stop execution here. The CookieController component rendered above 
            # will instantly read the cookie and trigger a rerun behind the scenes.
            st.stop()
            
        # If we waited and still have no user, they are truly logged out.
        _login_ui()
        st.stop()
        
    # Clear the wait flag for future runs now that we are logged in
    st.session_state.pop("cookie_waited", None)
    
    return user


def logout() -> None:
    """Clear the current session."""
    user = st.session_state.get(_SESSION_KEY)
    if user:
        try:
            db.update_session_token(user["id"], clear=True)
        except Exception:
            pass
            
    # Wipe the entire session state to ensure no cached states survive
    st.session_state.clear()
    
    try:
        controller = CookieController()
        controller.remove("auth_session_token")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

import re
import bcrypt

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    if not hashed:
        return False
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def _set_session_and_rerun(user: dict) -> None:
    """Store user in session, cookie, and URL, then rerun to enter app."""
    # Generate a secure session token
    token = db.update_session_token(user["id"])
    
    st.session_state[_SESSION_KEY] = user
    try:
        controller = CookieController()
        # Set cookie to expire in 5 days as requested by user
        expires = datetime.datetime.now() + datetime.timedelta(days=5)
        controller.set("auth_session_token", token, expires=expires)
    except Exception:
        pass
        
    st.query_params["session_token"] = token
    st.rerun()


def _do_login(email: str, password: str) -> None:
    email = email.strip().lower()
    if not email or not password:
        st.error("Please enter both email and password.")
        return
        
    user = db.get_user_by_email(email)
    if not user:
        st.error("Invalid email or password.")
        return
        
    if not user.get("hashed_password"):
        st.warning("Account missing password! Since this account was created during the preview phase, please go to 'Sign Up' and register again with the same email to set a secure password.")
        return
        
    if not verify_password(password, user["hashed_password"]):
        st.error("Invalid email or password.")
        return
        
    with st.spinner("Logging in..."):
        _set_session_and_rerun(user)


def _do_signup(email: str, name: str, password: str, confirm_password: str) -> None:
    email = email.strip().lower()
    name = name.strip()
    
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not email or not re.match(email_regex, email):
        st.error("Please enter a valid email address.")
        return
        
    if not password or len(password) < 6:
        st.error("Password must be at least 6 characters long.")
        return
        
    if password != confirm_password:
        st.error("Passwords do not match.")
        return
        
    with st.spinner("Creating account..."):
        hashed = hash_password(password)
        user = db.get_or_create_user(email, name=name, hashed_password=hashed)
        _set_session_and_rerun(user)


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
        .stTabs [data-baseweb="tab-list"] {
            justify-content: center;
        }
        </style>
        <div class="login-card">
            <h2>🧭 Learning Path Generator</h2>
            <p>Sign in or create an account to start generating personalized learning paths.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["Log In", "Sign Up"])
        
        with tab_login:
            login_email = st.text_input("Email", placeholder="you@example.com", key="login_email")
            login_pass = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
            if st.button("Log In →", use_container_width=True, type="primary", key="login_btn"):
                with st.spinner("Signing you in…"):
                    _do_login(login_email, login_pass)
                    
        with tab_signup:
            signup_name = st.text_input("Name", placeholder="Your Name", key="signup_name")
            signup_email = st.text_input("Email", placeholder="you@example.com", key="signup_email")
            signup_pass = st.text_input("Password", type="password", placeholder="At least 6 characters", key="signup_pass")
            signup_confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="signup_confirm")
            if st.button("Create Account →", use_container_width=True, type="primary", key="signup_btn"):
                with st.spinner("Creating account…"):
                    _do_signup(signup_email, signup_name, signup_pass, signup_confirm)
