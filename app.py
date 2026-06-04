"""
app.py — Multi-user Streamlit frontend for the Learning Path Generator.

Architecture:
  Login → Connect Integrations → Enter Goal → Generate → View Results

The app strictly separates concerns:
  - auth.py       handles session & login
  - composio_client.py handles OAuth connection flow
  - planner.py    handles LLM generation (structured JSON output)
  - actions.py    handles YouTube/Drive/Notion writes using user's account
  - db.py         persists user data, connections, and path history
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

import importlib
import auth
import composio_client as cc
import db
import actions
import planner
import schemas
import utils

importlib.reload(auth)
importlib.reload(cc)
importlib.reload(db)
importlib.reload(actions)
importlib.reload(planner)
importlib.reload(schemas)
importlib.reload(utils)

from actions import run_post_generation_actions
from planner import generate as generate_path
from schemas import learning_path_to_markdown
from utils import extract_video_ids_from_text, filter_available_videos, extract_video_ids_from_learning_path

load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_notion_page_id(page_id: str) -> str:
    """Extract a 32-character hex ID (with or without dashes) from user input."""
    if not page_id:
        return ""
    page_id = page_id.strip()
    
    # Try matching standard UUID format (8-4-4-4-12 hex chars)
    import re
    uuid_match = re.search(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
        page_id,
        re.IGNORECASE
    )
    if uuid_match:
        return uuid_match.group(1)
        
    # Match 32 hex characters in the string (common in Notion URLs)
    hex32_match = re.search(r"([0-9a-f]{32})", page_id, re.IGNORECASE)
    if hex32_match:
        return hex32_match.group(1)
        
    return page_id

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Learning Path Generator",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');
:root {
    --bg-dark: #050913;
    --card-bg: rgba(15, 23, 42, 0.6);
    --card-border: rgba(148, 163, 184, 0.1);
    --accent: #8b5cf6;
    --accent-glow: rgba(139, 92, 246, 0.25);
    --accent-2: #06b6d4;
    --accent-2-glow: rgba(6, 182, 212, 0.25);
    --text-muted: #94a3b8;
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, rgba(6,182,212,0.08), transparent 40%),
                radial-gradient(circle at 90% 10%, rgba(139,92,246,0.1), transparent 45%),
                radial-gradient(circle at 50% 80%, rgba(16,185,129,0.06), transparent 50%),
                #030712;
    font-family: 'Manrope', sans-serif;
    color: #f8fafc;
}

[data-testid="stSidebar"] > div:first-child {
    background: rgba(3, 7, 18, 0.85);
    backdrop-filter: blur(24px);
    border-right: 1px solid rgba(148, 163, 184, 0.08);
}

/* Hide Streamlit brand elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: rgba(3, 7, 18, 0.3);
}
::-webkit-scrollbar-thumb {
    background: rgba(139, 92, 246, 0.25);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(139, 92, 246, 0.5);
}

/* Hero */
.hero-card {
    border-radius: 20px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
    background: linear-gradient(135deg, rgba(139,92,246,0.65) 0%, rgba(6,182,212,0.45) 100%);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
    animation: float 14s ease-in-out infinite;
}
.hero-card h1 { 
    font-size: 2.3rem; 
    font-weight: 700;
    margin: 0 0 0.4rem 0; 
    letter-spacing: -0.02em;
    text-shadow: 0 2px 10px rgba(0,0,0,0.15);
}
.hero-card p  { 
    color: rgba(248,250,252,0.9); 
    margin: 0 0 0.8rem 0; 
    font-size: 1.05rem; 
    font-weight: 500;
}

/* Glass cards */
.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.3);
    margin-bottom: 1.2rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 25px 55px rgba(139, 92, 246, 0.12);
    border-color: rgba(139, 92, 246, 0.25);
}

/* Recent paths cards */
.recent-path-card {
    background: rgba(15, 23, 42, 0.3);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: all 0.25s ease;
}
.recent-path-card:hover {
    background: rgba(15, 23, 42, 0.45);
    border-color: rgba(139, 92, 246, 0.2);
}

/* Integration chips */
.int-row { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.5rem; }
.int-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.5rem 1.1rem; border-radius: 99px; font-size: 0.86rem;
    font-weight: 600; border: 1px solid transparent;
    transition: all 0.25s ease;
}
.int-chip.connected {
    background: rgba(16, 185, 129, 0.1); border-color: rgba(16, 185, 129, 0.3); color: #34d399;
}
.int-chip.disconnected {
    background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.25); color: #f87171;
}

/* Result chips */
.result-chip {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.55rem 1.2rem; border-radius: 99px; font-size: 0.9rem;
    font-weight: 600; border: 1px solid rgba(6, 182, 212, 0.35);
    background: rgba(6, 182, 212, 0.08); color: #e0f2fe;
    text-decoration: none; transition: all 0.25s ease;
}
.result-chip:hover { 
    border-color: rgba(139, 92, 246, 0.5); 
    background: rgba(139, 92, 246, 0.16); 
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.15);
}
.chip-row { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 1rem 0; }

/* Response block */
.response-block {
    background: rgba(3, 7, 18, 0.25); 
    border-radius: 14px; 
    padding: 1.4rem; 
    line-height: 1.8;
}
.response-block h1 {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    margin-top: 0.8rem !important;
    margin-bottom: 1.2rem !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12) !important;
    padding-bottom: 0.4rem !important;
}
.response-block h2 {
    font-size: 1.4rem !important;
    font-weight: 650 !important;
    color: #c084fc !important;
    margin-top: 1.6rem !important;
    margin-bottom: 0.9rem !important;
}
.response-block h3 {
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: #22d3ee !important;
    margin-top: 1.1rem !important;
    margin-bottom: 0.7rem !important;
}
.response-block ul, .response-block ol {
    margin-left: 1.3rem !important;
    margin-bottom: 1rem !important;
}
.response-block li {
    margin-bottom: 0.4rem !important;
}
.response-block strong {
    color: #ffffff !important;
}

/* Sidebar card */
.sb-card {
    background: rgba(15,23,42,0.4); border: 1px solid rgba(148,163,184,0.08);
    border-radius: 14px; padding: 1.1rem; margin-bottom: 1rem;
}

/* Inputs styling override */
div[data-testid="stTextInput"] input, div[data-testid="stChatInput"] textarea {
    background: rgba(3, 7, 18, 0.4) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.25s ease !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: rgba(139, 92, 246, 0.5) !important;
    box-shadow: 0 0 12px rgba(139, 92, 246, 0.2) !important;
}

/* Chat message override */
div[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.35) !important;
    border: 1px solid rgba(148, 163, 184, 0.08) !important;
    border-radius: 14px !important;
    margin-bottom: 0.8rem !important;
    padding: 1rem !important;
}

/* Buttons */
.stButton button {
    border-radius: 99px; 
    padding: 0.65rem 2rem;
    background: linear-gradient(135deg, #8b5cf6 0%, #06b6d4 100%);
    border: none; 
    color: #fff; 
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 10px 25px rgba(139, 92, 246, 0.22);
    transition: all 0.25s ease;
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 15px 30px rgba(139, 92, 246, 0.35);
    background: linear-gradient(135deg, #a78bfa 0%, #22d3ee 100%);
}

/* Secondary disconnect button styles */
.stButton button[key*="disconnect"] {
    background: rgba(239, 68, 68, 0.08) !important;
    border: 1px solid rgba(239, 68, 68, 0.25) !important;
    color: #f87171 !important;
    box-shadow: none !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.82rem !important;
}
.stButton button[key*="disconnect"]:hover {
    background: rgba(239, 68, 68, 0.16) !important;
    border-color: rgba(239, 68, 68, 0.4) !important;
}

/* Section label */
.section-label {
    margin: 2rem 0 0.6rem;
    text-transform: uppercase; 
    letter-spacing: 0.15em;
    font-size: 0.8rem;
    font-weight: 700;
    color: #64748b;
    border-left: 3px solid var(--accent);
    padding-left: 0.6rem;
}

/* Status strip */
.status-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin-bottom: 1.2rem;
}
.status-tile {
    background: rgba(15,23,42,0.4);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}
.status-tile .label {
    display: block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.3rem;
}
.status-tile .value {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f8fafc;
}

@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }

/* Responsiveness overrides */
@media (max-width: 900px) {
    .status-strip {
        grid-template-columns: 1fr;
        gap: 0.8rem;
    }
    .hero-card {
        padding: 1.8rem;
        margin-bottom: 1.2rem;
    }
    .hero-card h1 {
        font-size: 1.8rem;
    }
    .glass-card,
    .sb-card {
        padding: 1.2rem;
    }
}

@media (max-width: 480px) {
    [data-testid="stAppViewContainer"] {
        font-size: 0.95rem;
    }
    .hero-card h1 {
        font-size: 1.5rem;
    }
    .hero-card p {
        font-size: 0.9rem;
    }
    .stButton button {
        width: 100%;
        padding: 0.7rem 1rem;
    }
    .chip-row {
        flex-direction: column;
    }
    .result-chip {
        width: 100%;
        justify-content: center;
    }
}

/* Disable default Streamlit fading during rerun */
div[data-testid="stAppViewBlockContainer"] {
    opacity: 1 !important;
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Privacy policy route
# ---------------------------------------------------------------------------

params = st.query_params
if params.get("privacy") or params.get("privacy_policy"):
    try:
        with open("PRIVACY.md", "r", encoding="utf-8") as fh:
            st.markdown("# Privacy Policy\n" + fh.read())
    except Exception:
        st.header("Privacy Policy")
        st.write(
            "This application connects to your Google/YouTube and Notion accounts through "
            "Composio OAuth. No credentials are stored directly — only your Composio "
            "connected account IDs are stored in a local database."
        )
    st.stop()

# ---------------------------------------------------------------------------
# OAuth callback handling (query param: ?oauth_callback=provider&account_id=xxx)
# ---------------------------------------------------------------------------

def _handle_oauth_callback() -> None:
    """Process Composio OAuth redirect if present in query params."""
    provider = params.get("oauth_callback", "")
    account_id = params.get("account_id", "") or params.get("connected_account_id", "")
    if not provider or not account_id:
        return
    user = auth.get_current_user()
    if not user:
        return
    try:
        cc.handle_oauth_callback(user["id"], provider, account_id)
        st.success(f"✅ {provider.title()} connected successfully!")
        # Clear query params and rerun
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to save {provider} connection: {e}")


# ---------------------------------------------------------------------------
# Ensure DB is initialised on every cold start
# ---------------------------------------------------------------------------

db.init_db()

# ---------------------------------------------------------------------------
# Auth gate — must be logged in to proceed
# ---------------------------------------------------------------------------

user = auth.require_login()
user_id = user["id"]

_handle_oauth_callback()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _ss(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default

_ss("chat_history", [])
_ss("pending_goal", "")
_ss("generation_result", None)   # {'lp': LearningPath, 'markdown': str, 'actions': dict}
_ss("is_generating", False)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"### 👤 {user['name']}")
    st.caption(user["email"])
    if st.button("Sign out", key="signout_btn"):
        auth.logout()
        st.rerun()

    st.divider()

    # --- Model selection ---
    st.markdown("<div class='sb-card'>", unsafe_allow_html=True)
    st.markdown("**🧠 AI Model**")
    available_models = {
        "Gemini 2.5 Flash": "gemini-2.5-flash",
        "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
        "Mistral 7B Instruct (HF)": "mistralai/Mistral-7B-Instruct-v0.2",
        "Llama 3 8B Instruct (HF)": "NousResearch/Meta-Llama-3-8B-Instruct",
    }
    selected_model_label = st.selectbox(
        "Select model",
        options=list(available_models.keys()),
        key="model_select",
        label_visibility="collapsed",
    )
    selected_model = available_models[selected_model_label]
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # --- Integrations ---
    st.markdown("**🔗 Integrations**")
    conn_status = cc.get_connection_status(user_id)

    PROVIDERS = [
        ("youtube",     "YouTube",      "📺"),
        ("googledrive", "Google Drive", "📄"),
        ("notion",      "Notion",       "📝"),
    ]

    for provider, label, icon in PROVIDERS:
        connected = conn_status.get(provider, False)
        chip_class = "connected" if connected else "disconnected"
        status_text = "Connected" if connected else "Not connected"
        st.markdown(
            f"<div class='int-chip {chip_class}' style='width:100%; box-sizing:border-box;'>{icon} {label} — {status_text}</div>",
            unsafe_allow_html=True,
        )
        if not connected:
            if st.button(f"Connect {label}", key=f"connect_{provider}", use_container_width=True):
                try:
                    # Build redirect URL back to this app with oauth_callback param
                    app_url = os.getenv("APP_URL", "http://localhost:8501")
                    # Include user email to preserve session on redirect
                    user_email = user["email"]
                    redirect = f"{app_url}?oauth_callback={provider}&login_email={user_email}"
                    
                    oauth_url = cc.get_oauth_url(user_id, provider, redirect_url=redirect)
                    st.markdown(
                        f"<div style='margin-top: 5px; margin-bottom: 5px; text-align: center;'>"
                        f"<a href='{oauth_url}' target='_self' "
                        f"style='color:#22d3ee;text-decoration:underline;font-weight:bold;font-size:0.9rem;'>"
                        f"Click here to authorise {label} →</a>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption("This will authorize connection in this tab and redirect you back automatically.")
                    if provider == "googledrive":
                        st.caption("⚠️ Note: Make sure to check the box for full Google Drive file write access on the Google permission page.")
                    elif provider == "notion":
                        st.caption("⚠️ Note: Make sure to select all pages you want the generator to access on the Notion permission page.")
                except Exception as e:
                    st.error(f"Could not start OAuth for {label}: {e}")
        else:
            if st.button(f"Disconnect {label}", key=f"disconnect_{provider}", use_container_width=True):
                cc.disconnect(user_id, provider)
                st.rerun()
            
            if provider == "notion":
                current_parent_id = db.get_notion_parent_id(user_id) or ""
                new_parent_id = st.text_input(
                    "Parent Page ID",
                    value=current_parent_id,
                    key="notion_parent_id_input",
                    help="Open any Notion page in your browser and copy the 32-character ID from the URL. This is required because Notion does not allow creating pages at the root level."
                )
                sanitized_parent_id = sanitize_notion_page_id(new_parent_id)
                if sanitized_parent_id != current_parent_id:
                    db.save_notion_parent_id(user_id, sanitized_parent_id)
                    st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

    st.divider()
    st.caption("Connect services once — future prompts run automatically.")

# ---------------------------------------------------------------------------
# Main area — Hero
# ---------------------------------------------------------------------------

st.markdown(
    f"""
    <div class="hero-card">
        <h1>🧭 Learning Path Generator</h1>
        <p>Craft binge-worthy study plans, YouTube playlists, and docs — all from one prompt.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

connected_count = sum(1 for value in conn_status.values() if value)
st.markdown(
    f"""
    <div class="status-strip">
        <div class="status-tile">
            <span class="label">Integrations</span>
            <div class="value">{connected_count}/3 connected</div>
        </div>
        <div class="status-tile">
            <span class="label">Output</span>
            <div class="value">Structured plan + exports</div>
        </div>
        <div class="status-tile">
            <span class="label">Execution</span>
            <div class="value">Composio-hosted tools</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Connection status banner
any_connected = any(conn_status.values())
if not any_connected:
    st.info(
        "👈 Connect at least one integration in the sidebar to enable automatic playlist "
        "and document creation. You can still generate a learning path without any integrations."
    )

# ---------------------------------------------------------------------------
# Goal builder chat
# ---------------------------------------------------------------------------

st.markdown("<div class='section-label'>Goal Builder</div>", unsafe_allow_html=True)

for msg in st.session_state.chat_history[-6:]:
    with st.chat_message(msg.get("role", "user")):
        st.markdown(msg.get("content", ""))

chat_prompt = st.chat_input("Describe what you want to learn (e.g. 'Learn Python in 10 days')")
if chat_prompt:
    st.session_state.chat_history.append({"role": "user", "content": chat_prompt})
    st.session_state.pending_goal = chat_prompt
    st.session_state.generation_result = None
    st.rerun()

user_goal = st.session_state.pending_goal.strip()
# ---------------------------------------------------------------------------
# Generate button
# ---------------------------------------------------------------------------

progress_placeholder = st.empty()
status_placeholder = st.empty()

if st.button(
    "⚡ Generate Learning Path",
    type="primary",
    disabled=st.session_state.is_generating or not user_goal,
    key="generate_btn",
):
    st.session_state.is_generating = True
    st.session_state.generation_result = None
    steps: list[str] = []

    def _progress(msg: str) -> None:
        steps.append(msg)
        steps_html = "".join([f"<li>{s}</li>" for s in steps[-5:]])
        progress_placeholder.markdown(
            f"""
            <style>
            /* Hide other elements during loading */
            [data-testid="stSidebar"],
            .hero-card,
            .status-strip,
            .section-label,
            .glass-card,
            .recent-path-card,
            .stButton,
            [data-testid="stChatInput"],
            [data-testid="stChatMessage"] {{
                display: none !important;
            }}
            
            /* Standalone Loader Card */
            .custom-loader-card-standalone {{
                background: rgba(15, 23, 42, 0.85) !important;
                border: 1px solid rgba(139, 92, 246, 0.25) !important;
                border-radius: 24px !important;
                padding: 3rem 2.5rem !important;
                text-align: center !important;
                box-shadow: 0 30px 60px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05) !important;
                max-width: 480px !important;
                margin: 15vh auto !important;
                backdrop-filter: blur(16px) !important;
                -webkit-backdrop-filter: blur(16px) !important;
                animation: scaleUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
            }}
            .spinner-ring {{
                width: 60px !important;
                height: 60px !important;
                border: 4px solid rgba(139, 92, 246, 0.15) !important;
                border-top: 4px solid #8b5cf6 !important;
                border-radius: 50% !important;
                margin: 0 auto 1.8rem !important;
                animation: spin 0.8s linear infinite !important;
            }}
            .loader-steps {{
                list-style-type: none !important;
                padding-left: 0 !important;
                margin: 0 auto !important;
                max-width: 340px !important;
                text-align: left !important;
            }}
            .loader-steps li {{
                color: rgba(148, 163, 184, 0.6) !important;
                font-size: 0.9rem !important;
                margin-bottom: 0.5rem !important;
                display: flex !important;
                align-items: center !important;
                gap: 0.6rem !important;
                line-height: 1.4 !important;
            }}
            .loader-steps li::before {{
                content: "✓" !important;
                color: #10b981 !important;
                font-weight: bold !important;
            }}
            .loader-steps li:last-child {{
                color: #22d3ee !important;
                font-weight: 600 !important;
                animation: pulse 1.5s infinite !important;
            }}
            .loader-steps li:last-child::before {{
                content: "●" !important;
                color: #22d3ee !important;
                animation: blink 1s infinite !important;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            @keyframes scaleUp {{
                from {{ transform: scale(0.96); opacity: 0; }}
                to {{ transform: scale(1); opacity: 1; }}
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.6; }}
            }}
            @keyframes blink {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.3; }}
            }}
            </style>
            
            <div class="custom-loader-card-standalone">
                <div class="spinner-ring"></div>
                <h3 style="color:#ffffff; margin-top:0; margin-bottom:1.5rem; font-weight:700; font-size:1.45rem; letter-spacing:-0.02em;">🧭 Designing Your Path</h3>
                <ul class="loader-steps">
                    {steps_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Trigger initial loader state immediately
    _progress(f"Initializing with {selected_model_label}…")

    try:
        # --- Planning layer ---
        learning_path = generate_path(
            user_goal=user_goal,
            model_name=selected_model,
            progress_callback=_progress,
        )

        # --- Video validation ---
        valid_ids = []
        if conn_status.get("youtube"):
            _progress("Extracting and validating YouTube video IDs…")
            raw_ids = extract_video_ids_from_learning_path(learning_path)
            if raw_ids:
                valid_ids, invalid_ids = filter_available_videos(raw_ids)
                if invalid_ids:
                    _progress(f"Filtered {len(invalid_ids)} unavailable video(s).")
            else:
                _progress("No YouTube video IDs found in generated content.")

        # --- Generate Markdown with updated URLs ---
        _progress("Converting to markdown…")
        markdown_text = learning_path_to_markdown(learning_path)

        # --- Action layer ---
        _progress("Running post-generation actions…")
        action_results = run_post_generation_actions(
            user_id=user_id,
            goal=user_goal,
            markdown=markdown_text,
            video_ids=valid_ids,
            connection_status=conn_status,
            progress_callback=_progress,
        )

        # --- Rebuild markdown with real URLs ---
        playlist_url = None
        doc_url = None
        notion_url = None

        def _action_success(name: str) -> bool:
            action = action_results.get(name) or {}
            return bool(action.get("success"))

        if _action_success("playlist"):
            playlist_url = (action_results.get("playlist") or {}).get("playlist_url")
            _progress(f"Playlist created: {playlist_url}")

        if _action_success("google_doc"):
            doc_url = (action_results.get("google_doc") or {}).get("doc_url")
            _progress(f"Google Doc created: {doc_url}")

        if _action_success("notion_page"):
            notion_url = (action_results.get("notion_page") or {}).get("page_url")
            _progress(f"Notion page created: {notion_url}")

        final_markdown = learning_path_to_markdown(
            learning_path,
            playlist_url=playlist_url,
            doc_url=doc_url,
            notion_url=notion_url,
        )

        # --- Persist to DB ---
        db.save_learning_path(
            user_id=user_id,
            goal=user_goal,
            playlist_url=playlist_url,
            google_doc_url=doc_url,
            notion_url=notion_url,
            markdown=final_markdown,
        )

        st.session_state.generation_result = {
            "lp": learning_path,
            "markdown": final_markdown,
            "actions": action_results,
            "playlist_url": playlist_url,
            "doc_url": doc_url,
            "notion_url": notion_url,
        }
        _progress("✅ All done!")
        st.rerun()

    except Exception as e:
        st.error(f"Generation failed: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
    finally:
        st.session_state.is_generating = False
        progress_placeholder.empty()

result = st.session_state.generation_result
if result:
    st.markdown("<div class='section-label'>Your Learning Path</div>", unsafe_allow_html=True)

    # Action result chips
    chip_html = ""
    if result.get("playlist_url"):
        chip_html += (
            f"<a class='result-chip' href='{result['playlist_url']}' target='_blank'>"
            f"🎧 YouTube Playlist</a>"
        )
    if result.get("doc_url"):
        chip_html += (
            f"<a class='result-chip' href='{result['doc_url']}' target='_blank'>"
            f"📄 Google Doc</a>"
        )
    if result.get("notion_url"):
        chip_html += (
            f"<a class='result-chip' href='{result['notion_url']}' target='_blank'>"
            f"📝 Notion Page</a>"
        )
    if chip_html:
        st.markdown(f"<div class='chip-row'>{chip_html}</div>", unsafe_allow_html=True)

    # Show action errors (non-fatal)
    actions = result.get("actions", {})
    for key, label in [("playlist", "YouTube Playlist"), ("google_doc", "Google Doc"), ("notion_page", "Notion Page")]:
        action = actions.get(key)
        if action and not action.get("success") and action.get("error"):
            err_msg = action["error"]
            err_msg_lower = err_msg.lower()
            if key == "notion_page" and ("parent id" in err_msg_lower or "no page or database" in err_msg_lower or "not found" in err_msg_lower or "uuid" in err_msg_lower):
                st.error(
                    f"🔴 **Notion Parent Page Error**: Parent page not found or inaccessible.\n\n"
                    f"**Details**: `{err_msg}`\n\n"
                    f"**How to fix this:**\n"
                    f"1. **Share the Page with Integration**: Open the page you want to use as parent in Notion, click the **Share** button in the top-right corner, and verify that the integration (Composio or Learning Path Generator) is selected/added as a connection.\n"
                    f"2. **Check Parent ID**: Make sure the Parent Page ID entered in the sidebar is correct. You can copy the entire URL of the Notion page and paste it into the sidebar input field; the app will automatically extract the ID.\n"
                    f"3. **Reconnect Notion**: If the page is still not found, disconnect and reconnect Notion in the sidebar, ensuring you select the correct workspace and check the boxes for the pages you want the generator to access."
                )
            elif "insufficient authentication scopes" in err_msg_lower or "permission" in err_msg_lower or "forbidden" in err_msg_lower or "scope" in err_msg_lower:
                if key == "google_doc":
                    st.error(
                        f"🔴 **Google Doc Integration Error**: Insufficient Permissions.\n\n"
                        f"To fix this, please **Disconnect Google Drive** in the sidebar and click **Connect Google Drive** again. "
                        f"When Google displays the OAuth consent screen, **you MUST check the box** that says "
                        f"\"See, edit, create, and delete all your Google Drive files\" to allow the generator to save documents."
                    )
                elif key == "notion_page":
                    st.error(
                        f"🔴 **Notion Integration Error**: Insufficient Access.\n\n"
                        f"To fix this, please **Disconnect Notion** in the sidebar and click **Connect Notion** again. "
                        f"Make sure to select the workspace and pages you want the generator to access during Notion's OAuth authorization flow."
                    )
                else:
                    st.error(f"🔴 **{label} Error**: {err_msg}")
            else:
                st.warning(f"⚠️ {label}: {err_msg}")

    # Learning path content
    with st.container():
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='response-block'>", unsafe_allow_html=True)
        st.markdown(result["markdown"])
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Download button
    st.download_button(
        label="⬇️ Download Markdown",
        data=result["markdown"],
        file_name="learning_path.md",
        mime="text/markdown",
        key="download_md",
    )

# ---------------------------------------------------------------------------
# History panel
# ---------------------------------------------------------------------------

recent_paths = db.get_user_paths(user_id, limit=5)
if recent_paths:
    with st.expander("📋 Recent Learning Paths"):
        for path in recent_paths:
            with st.container():
                st.markdown("<div class='recent-path-card'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([2.5, 2, 1.2])
                with col1:
                    st.markdown(f"**🎯 Goal:**  \n{path['goal']}")
                with col2:
                    links = []
                    if path.get("playlist_url"):
                        links.append(f"[📺 YouTube Playlist]({path['playlist_url']})")
                    if path.get("google_doc_url"):
                        links.append(f"[📄 Google Doc]({path['google_doc_url']})")
                    if path.get("notion_url"):
                        links.append(f"[📝 Notion Page]({path['notion_url']})")
                    if links:
                        st.markdown("  \n".join(links))
                    else:
                        st.caption("No export links generated")
                with col3:
                    if st.button("👁️ View Plan", key=f"view_path_{path['id']}", use_container_width=True):
                        st.session_state.generation_result = {
                            "lp": None,
                            "markdown": path.get("markdown") or f"# Learning Path: {path['goal']}\n\n*Plan details not available. Please generate a new path to view full details.*",
                            "actions": {},
                            "playlist_url": path.get("playlist_url"),
                            "doc_url": path.get("google_doc_url"),
                            "notion_url": path.get("notion_url"),
                        }
                        st.session_state.pending_goal = path["goal"]
                        st.rerun()
                    st.caption(f"<div style='text-align:center; margin-top:0.2rem;'>📅 {path['created_at'][:10]}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
