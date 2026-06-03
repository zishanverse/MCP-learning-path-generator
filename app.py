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
    --card-bg: rgba(15, 23, 42, 0.75);
    --card-border: rgba(226, 232, 240, 0.2);
    --accent: #a855f7;
    --accent-2: #22d3ee;
    --text-muted: #cbd5f5;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, rgba(56,189,248,0.12), transparent 35%),
                radial-gradient(circle at 80% 0%, rgba(168,85,247,0.15), transparent 40%),
                radial-gradient(circle at 0% 80%, rgba(34,211,238,0.14), transparent 45%),
                #050913;
    font-family: 'Manrope', sans-serif;
    color: #f8fafc;
}
[data-testid="stSidebar"] > div:first-child {
    background: rgba(5, 8, 19, 0.9);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(148, 163, 184, 0.18);
}
/* Hero */
.hero-card {
    border-radius: 24px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.6rem;
    background: linear-gradient(135deg, rgba(124,58,237,0.7), rgba(14,165,233,0.55));
    box-shadow: 0 25px 70px rgba(15, 23, 42, 0.5);
    animation: float 14s ease-in-out infinite;
}
.hero-card h1 { font-size: 2.1rem; margin: 0 0 0.3rem 0; }
.hero-card p  { color: rgba(248,250,252,0.88); margin: 0 0 1rem 0; font-size: 1rem; }
/* Glass cards */
.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 20px 50px rgba(2,6,23,0.5);
    margin-bottom: 1.2rem;
}
/* Integration chips */
.int-row { display: flex; flex-wrap: wrap; gap: 0.6rem; margin-top: 0.5rem; }
.int-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.45rem 1rem; border-radius: 999px; font-size: 0.88rem;
    font-weight: 600; border: 1px solid transparent;
}
.int-chip.connected {
    background: rgba(34,197,94,0.12); border-color: rgba(34,197,94,0.4); color: #86efac;
}
.int-chip.disconnected {
    background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.35); color: #fca5a5;
}
/* Result chips */
.result-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.55rem 1.1rem; border-radius: 999px; font-size: 0.9rem;
    font-weight: 600; border: 1px solid rgba(34,211,238,0.45);
    background: rgba(34,211,238,0.1); color: #e0f2fe;
    text-decoration: none; transition: all 0.2s;
}
.result-chip:hover { border-color: rgba(124,58,237,0.55); background: rgba(124,58,237,0.18); }
.chip-row { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 1rem 0; }
/* Response block */
.response-block {
    background: rgba(15,23,42,0.5); border: 1px solid rgba(148,163,184,0.2);
    border-radius: 16px; padding: 1.2rem 1.4rem; line-height: 1.7;
}
/* Sidebar card */
.sb-card {
    background: rgba(15,23,42,0.5); border: 1px solid rgba(148,163,184,0.2);
    border-radius: 16px; padding: 1.1rem; margin-bottom: 1rem;
}
/* Buttons */
.stButton button {
    border-radius: 999px; padding: 0.8rem 2rem;
    background: linear-gradient(120deg, #7c3aed, #22d3ee);
    border: none; color: #fff; font-weight: 600;
    box-shadow: 0 12px 30px rgba(34,211,238,0.28);
}
/* Section label */
.section-label {
    margin: 1.6rem 0 0.5rem;
    text-transform: uppercase; letter-spacing: 0.15em;
    font-size: 0.85rem; color: #94a3b8;
}
.status-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
    margin-bottom: 1.2rem;
}
.status-tile {
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(148,163,184,0.2);
    border-radius: 16px;
    padding: 0.95rem 1rem;
    box-shadow: 0 18px 35px rgba(2,6,23,0.35);
}
.status-tile .label {
    display: block;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.35rem;
}
.status-tile .value {
    font-size: 1rem;
    font-weight: 700;
    color: #f8fafc;
}
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-5px)} }
@keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }

@media (max-width: 900px) {
    .status-strip {
        grid-template-columns: 1fr;
    }
    .hero-card {
        padding: 1.5rem 1.3rem;
    }
    .hero-card h1 {
        font-size: 1.7rem;
    }
    .glass-card,
    .sb-card {
        padding: 1rem;
    }
}

@media (max-width: 640px) {
    [data-testid="stAppViewContainer"] {
        font-size: 0.96rem;
    }
    .stButton button {
        width: 100%;
        padding: 0.8rem 1rem;
    }
    .chip-row {
        flex-direction: column;
    }
    .result-chip {
        width: 100%;
        justify-content: center;
    }
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
            f"<div class='int-chip {chip_class}'>{icon} {label} — {status_text}</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns([1, 1])
        if not connected:
            with cols[0]:
                if st.button(f"Connect {label}", key=f"connect_{provider}"):
                    try:
                        # Build redirect URL back to this app with oauth_callback param
                        app_url = os.getenv("APP_URL", "http://localhost:8501")
                        redirect = f"{app_url}?oauth_callback={provider}"
                        oauth_url = cc.get_oauth_url(user_id, provider, redirect_url=redirect)
                        st.markdown(
                            f"<a href='{oauth_url}' target='_blank' "
                            f"style='color:#22d3ee;text-decoration:underline;'>"
                            f"Click here to authorise {label} →</a>",
                            unsafe_allow_html=True,
                        )
                        if provider == "googledrive":
                            st.caption("⚠️ Note: Make sure to check the box for full Google Drive file write access on the Google permission page.")
                        elif provider == "notion":
                            st.caption("⚠️ Note: Make sure to select all pages you want the generator to access on the Notion permission page.")
                    except Exception as e:
                        st.error(f"Could not start OAuth for {label}: {e}")
        else:
            with cols[0]:
                if st.button(f"Disconnect", key=f"disconnect_{provider}"):
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
                if new_parent_id != current_parent_id:
                    db.save_notion_parent_id(user_id, new_parent_id)
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
        progress_placeholder.markdown(
            "\n".join([f"- {s}" for s in steps[-5:]]),
        )

    try:
        with st.spinner(f"Generating with {selected_model_label}…"):
            # --- Planning layer ---
            learning_path = generate_path(
                user_goal=user_goal,
                model_name=selected_model,
                progress_callback=_progress,
            )

            # --- Video validation ---
            _progress("Extracting and validating YouTube video IDs…")
            raw_ids = extract_video_ids_from_learning_path(learning_path)
            if raw_ids:
                valid_ids, invalid_ids = filter_available_videos(raw_ids)
                if invalid_ids:
                    _progress(f"Filtered {len(invalid_ids)} unavailable video(s).")
            else:
                valid_ids = []
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

    except Exception as e:
        st.error(f"Generation failed: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
    finally:
        st.session_state.is_generating = False
        progress_placeholder.empty()
        st.rerun()

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

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
            if "insufficient authentication scopes" in err_msg.lower() or "permission" in err_msg.lower() or "forbidden" in err_msg.lower() or "scope" in err_msg.lower():
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
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(path["goal"])
            with col2:
                if path.get("playlist_url"):
                    st.markdown(f"[Playlist]({path['playlist_url']})")
            with col3:
                st.caption(path["created_at"][:10])
