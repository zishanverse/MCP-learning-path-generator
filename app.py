import os
import streamlit as st
from dotenv import load_dotenv
from utils import run_agent_sync

try:
    from scripts.get_youtube_token import obtain_youtube_token
except Exception:
    obtain_youtube_token = None

load_dotenv()

st.set_page_config(page_title="MCP POC", page_icon="🤖", layout="wide")

custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');
:root {
    --bg-dark: #050913;
    --card-bg: rgba(15, 23, 42, 0.75);
    --card-border: rgba(226, 232, 240, 0.2);
    --accent: #a855f7;
    --accent-2: #22d3ee;
    --text-muted: #cbd5f5;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 20% 20%, rgba(56,189,248,0.15), transparent 35%),
                radial-gradient(circle at 80% 0%, rgba(168,85,247,0.18), transparent 40%),
                radial-gradient(circle at 0% 80%, rgba(34,211,238,0.18), transparent 45%),
                #050913;
    font-family: 'Manrope', sans-serif;
    color: #f8fafc;
}
[data-testid="stSidebar"] > div:first-child {
    background: rgba(5, 8, 19, 0.85);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(148, 163, 184, 0.2);
}
.hero-card {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 2.6rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(135deg, rgba(124,58,237,0.75), rgba(14,165,233,0.6));
    box-shadow: 0 30px 80px rgba(15, 23, 42, 0.55);
    animation: float 14s ease-in-out infinite;
}
.hero-content {
    display: flex;
    flex-wrap: wrap;
    gap: 2rem;
    align-items: center;
    position: relative;
    z-index: 2;
}
.hero-text h1 {
    font-size: 2.4rem;
    margin-bottom: 0.5rem;
}
.hero-text p {
    color: rgba(248, 250, 252, 0.9);
    margin-bottom: 1rem;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}
.hero-badge {
    background: rgba(15,23,42,0.35);
    border: 1px solid rgba(248, 250, 252, 0.25);
    padding: 0.35rem 0.9rem;
    border-radius: 999px;
    font-size: 0.9rem;
    backdrop-filter: blur(6px);
}
.hero-badge.success {
    border-color: rgba(34,197,94,0.4);
    color: #bbf7d0;
}
.hero-badge.warn {
    border-color: rgba(248,113,113,0.5);
    color: #fecaca;
}
.sidebar-card {
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 20px;
    padding: 1.3rem 1.1rem;
    margin-bottom: 1.1rem;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
}
.sidebar-card .card-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.sidebar-card .card-desc {
    color: rgba(226,232,240,0.75);
    font-size: 0.9rem;
    margin-bottom: 0.8rem;
}
.sidebar-card [data-baseweb="select"] {
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.45);
    background: rgba(15,23,42,0.65);
}
.sidebar-card [data-baseweb="select"] div {
    color: #f8fafc;
}
.model-pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.8rem;
}
.model-pill {
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,0.4);
    background: rgba(148,163,184,0.12);
    font-size: 0.85rem;
}
.hero-visual {
    position: relative;
    width: 260px;
    height: 260px;
}
.orb {
    position: absolute;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    filter: blur(0px);
    opacity: 0.7;
}
.orb-one { background: radial-gradient(circle, #38bdf8, transparent 60%); top: 0; right: 10px; animation: pulse 6s infinite; }
.orb-two { background: radial-gradient(circle, #c084fc, transparent 65%); bottom: 0; left: 0; animation: pulse 8s infinite; }
.hero-stat {
    position: absolute;
    top: 52%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 170px;
    height: 170px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, rgba(34,211,238,0.9), rgba(124,58,237,0.55));
    border: 1px solid rgba(255,255,255,0.35);
    box-shadow: 0 25px 50px rgba(14,165,233,0.35);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #0f172a;
    backdrop-filter: blur(10px);
    animation: float 6s ease-in-out infinite;
}
.hero-stat-label {
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: rgba(15, 23, 42, 0.7);
    margin-bottom: 0.15rem;
}
.hero-stat-value {
    font-size: 2.8rem;
    font-weight: 600;
    line-height: 1;
}
.hero-stat-sub {
    font-size: 0.9rem;
    color: rgba(15,23,42,0.75);
}
.glass-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 20px;
    padding: 1.5rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 25px 55px rgba(2,6,23,0.6);
}
.tip-card ul {
    margin: 0;
    padding-left: 1.1rem;
    color: var(--text-muted);
}
.section-title {
    margin-top: 2rem;
    margin-bottom: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 0.95rem;
    color: #94a3b8;
}
[data-testid="stChatMessage"] {
    border-radius: 18px;
    border: 1px solid rgba(226, 232, 240, 0.15);
    background: rgba(255,255,255,0.04);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    animation: fadeIn 0.45s ease;
}
[data-testid="stChatInput"] textarea {
    border-radius: 18px !important;
    border: 1px solid rgba(148,163,184,0.4) !important;
    background: rgba(15,23,42,0.6) !important;
    color: #f8fafc !important;
    min-height: 140px !important;
    font-size: 1.05rem !important;
    line-height: 1.5 !important;
    padding: 1.15rem !important;
}
.stButton button {
    border-radius: 999px;
    padding: 0.85rem 2.4rem;
    background: linear-gradient(120deg, #7c3aed, #22d3ee);
    border: none;
    color: #fff;
    font-weight: 600;
    box-shadow: 0 15px 35px rgba(34,211,238,0.3);
    animation: glow 6s ease-in-out infinite;
}
.stTabs [data-baseweb="tab-list"] {
    justify-content: flex-start;
    border-bottom: 1px solid rgba(148,163,184,0.25);
}
.stTabs [data-baseweb="tab-list"] button {
    border-radius: 999px;
    margin-right: 0.5rem;
    color: #cbd5f5;
}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.5);
    color: #fff;
}
.response-block {
    background: rgba(15,23,42,0.55);
    border: 1px solid rgba(148,163,184,0.25);
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.6rem;
    line-height: 1.65;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
}
.playlist-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1rem;
}
.playlist-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.55rem 1rem;
    border-radius: 999px;
    border: 1px solid rgba(34,211,238,0.45);
    background: rgba(34,211,238,0.12);
    color: #e0f2fe;
    text-decoration: none;
    font-weight: 600;
    font-size: 0.9rem;
    transition: all 0.2s ease;
}
.playlist-chip:hover {
    border-color: rgba(124,58,237,0.6);
    background: rgba(124,58,237,0.2);
}
@keyframes float { 0%,100% { transform: translateY(0px);} 50% { transform: translateY(-6px);} }
@keyframes pulse { 0% { transform: scale(1); opacity: 0.8;} 50% { transform: scale(1.2); opacity: 0.4;} 100% { transform: scale(1); opacity: 0.8;} }
@keyframes spin { from { transform: translate(-50%, -50%) rotate(0deg);} to { transform: translate(-50%, -50%) rotate(360deg);} }
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px);} to { opacity: 1; transform: translateY(0);} }
@keyframes glow { 0% { box-shadow: 0 15px 35px rgba(34,211,238,0.35);} 50% { box-shadow: 0 25px 45px rgba(124,58,237,0.45);} 100% { box-shadow: 0 15px 35px rgba(34,211,238,0.35);} }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("Model Context Protocol(MCP) - Learning Path Generator")

# Quick privacy policy route: if ?privacy=1 is present, show the policy and exit
params = st.query_params
if params.get("privacy") or params.get("privacy_policy"):
    # Load privacy policy from disk if present, otherwise show built-in text
    try:
        with open("PRIVACY.md", "r", encoding="utf-8") as fh:
            md = fh.read()
        st.markdown("# Privacy Policy\n" + md)
    except Exception:
        st.header("Privacy Policy")
        st.write("This application collects only the minimum data required to operate (OAuth tokens for YouTube when you opt in). See repository README for details.")
        st.write("For more information contact the application owner.")
    st.stop()

# Initialize session state for progress and model results
if 'current_step' not in st.session_state:
    st.session_state.current_step = ""
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'last_section' not in st.session_state:
    st.session_state.last_section = ""
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pending_goal' not in st.session_state:
    st.session_state.pending_goal = ""


# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")

# Define the available models and their API names
available_models = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    # "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
    "Claude 3 Sonnet": "claude-3-sonnet-20240229",
    "Mistral Large": "mistral-large-latest",
    "Perplexity": "perplexity-ai/llama-3-8b-instruct",
}

with st.sidebar.container():
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>AI Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-desc'>Pick the brains you want to pit against each other.</div>", unsafe_allow_html=True)
    selected_model_names = st.multiselect(
        "Select AI Models for Comparison:",
        options=list(available_models.keys()),
        default=["Gemini 2.5 Flash", "Mistral Large"],
        key="model_compare_select"
    )
    if selected_model_names:
        chips = "".join([f"<span class='model-pill'>{name}</span>" for name in selected_model_names])
        st.markdown(f"<div class='model-pill-row'>{chips}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

selected_models = [available_models[name] for name in selected_model_names]

# HIGHLIGHT: Removed manual API key input

# Composio Integration Selection
st.sidebar.subheader("Composio Integrations")
st.sidebar.caption("✅ Using Composio MCP (mcp.composio.com)")

# YouTube OAuth connection helper
token_file_env = os.getenv("YOUTUBE_OAUTH_TOKEN_FILE", "tokens/youtube_token.json")
token_file_path = os.path.abspath(os.path.expanduser(token_file_env))
client_secrets_env = os.getenv("YOUTUBE_OAUTH_CLIENT_SECRETS_FILE", "youtube_client_secrets.json")
client_secrets_path = os.path.abspath(os.path.expanduser(client_secrets_env))
oauth_port = os.getenv("YOUTUBE_OAUTH_LOCAL_PORT", "8765")
youtube_connected = os.path.exists(token_file_path)

if youtube_connected:
    st.sidebar.success("YouTube OAuth connected ✅")
else:
    st.sidebar.warning("YouTube not connected. Connect to enable playlist creation.")

connect_button_disabled = (obtain_youtube_token is None) or youtube_connected
connect_button_label = "YouTube Connected" if youtube_connected else "Connect YouTube via OAuth"
if st.sidebar.button(connect_button_label, disabled=connect_button_disabled):
    if obtain_youtube_token is None:
        st.sidebar.error("OAuth helper not available. Please run scripts/get_youtube_token.py manually.")
    else:
        try:
            with st.spinner("Launching OAuth flow..."):
                obtain_youtube_token(token_file=token_file_path, client_secrets=client_secrets_path, port=int(oauth_port))
            st.sidebar.success("YouTube connected! Token saved.")
            youtube_connected = True
        except Exception as oauth_err:
            st.sidebar.error(f"OAuth flow failed: {oauth_err}")

# Secondary tool selection
secondary_tool = st.sidebar.radio(
    "Select Secondary Tool:",
    ["None", "Drive", "Notion"],
    index=0
)
st.sidebar.caption("Drive/Notion reuse your Composio connection when toggled.")

# Set integration flags
use_youtube = True  # Always enabled
use_drive = (secondary_tool == "Drive")
use_notion = (secondary_tool == "Notion")

integration_label = secondary_tool if secondary_tool != "None" else "Core"
youtube_badge = "YouTube linked" if youtube_connected else "Connect YouTube"
recent_prompts = max(len(st.session_state.chat_history), 1)

hero_html = f"""
<div class="hero-card">
    <div class="hero-content">
        <div class="hero-text">
            <p class="eyebrow">Context-aware learning journeys</p>
            <h1>Learning Path Copilot</h1>
            <p>Craft binge-worthy study plans, ready-to-share docs, and playlists of fresh videos—all from one prompt.</p>
            <div class="hero-badges">
                <span class="hero-badge">🧠 {len(selected_model_names)} models active</span>
                <span class="hero-badge">🧩 {integration_label} tools</span>
                <span class="hero-badge {'success' if youtube_connected else 'warn'}">{youtube_badge}</span>
            </div>
        </div>
        <div class="hero-visual">
            <div class="orb orb-one"></div>
            <div class="orb orb-two"></div>
            <div class="hero-stat">
                <span class="hero-stat-label">Prompt count</span>
                <span class="hero-stat-value">{recent_prompts}</span>
                <span class="hero-stat-sub">today</span>
            </div>
        </div>
    </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

tip_card = """
<div class="glass-card tip-card">
    <div class="tip-title" style="font-weight:600;margin-bottom:0.6rem;">Flow</div>
    <ul>
        <li>Pick the models you want to compare and toggle optional Drive/Notion syncing.</li>
        <li>Describe a goal with the timeframe you have in mind—single prompt, conversation-style.</li>
        <li>We handle doc creation & playlist curation automatically, surfacing links in the summary.</li>
    </ul>
</div>
"""
st.markdown(tip_card, unsafe_allow_html=True)

st.markdown("<div class='section-title'>Goal Builder</div>", unsafe_allow_html=True)
for message in st.session_state.chat_history[-6:]:
    with st.chat_message(message.get("role", "user")):
        st.markdown(message.get("content", ""))

chat_prompt = st.chat_input("Describe what you want to learn (e.g., 3-day data science sprint)")
if chat_prompt:
    st.session_state.chat_history.append({"role": "user", "content": chat_prompt})
    st.session_state.pending_goal = chat_prompt
    st.rerun()

user_goal = st.session_state.pending_goal.strip()

# Progress area
progress_container = st.container()
progress_bar = st.empty()

# Update progress function to show which model is running
def update_progress(message: str, model_tag: str = ""):
    """Update progress in the Streamlit UI"""
    st.session_state.current_step = f"[{model_tag}] {message}" if model_tag else message
    
    # Determine section and update progress
    if "Setting up agent with tools" in message:
        section = "Setup"
        st.session_state.progress = 0.1
    elif "Added Google Drive integration" in message or "Added Notion integration" in message:
        section = "Integration"
        st.session_state.progress = 0.2
    elif "Creating AI agent" in message:
        section = "Setup"
        st.session_state.progress = 0.3
    elif "Generating your learning path" in message:
        section = "Generation"
        st.session_state.progress = 0.5
    elif "Learning path generation complete" in message:
        section = "Complete"
        st.session_state.progress = 1.0
        st.session_state.is_generating = False
    else:
        section = st.session_state.last_section or "Progress"
    
    st.session_state.last_section = section
    
    # Show progress bar
    progress_bar.progress(st.session_state.progress)
    
    # Update progress container with current status
    with progress_container:
        if section != st.session_state.last_section and section != "Complete":
            st.write(f"**{section}**")
        
        if "Learning path generation complete!" in message:
            st.success(f"All steps completed for {model_tag}! 🎉")
        else:
            prefix = "✓" if st.session_state.progress >= 0.5 else "→"
            st.write(f"{prefix} {st.session_state.current_step}")


# Generate Learning Path button
if st.button("Generate Learning Path", type="primary", disabled=st.session_state.is_generating):
    # Validate Composio API key is set
    composio_api_key = os.getenv("COMPOSIO_API_KEY")
    
    if not selected_models:
        st.error("Please select at least one AI model for comparison.")
    elif not composio_api_key:
        st.error("⚠️ COMPOSIO_API_KEY not found! Please set it in your .env file.")
        st.info("💡 Create a .env file with: COMPOSIO_API_KEY=your_key_here")
    elif not user_goal:
        st.warning("Please enter your learning goal.")
    else:
        st.session_state.is_generating = True
        st.session_state.model_results = {}
        st.session_state.current_step = ""
        st.session_state.progress = 0
        st.session_state.last_section = ""

        # This will now just run the process without displaying results here
        for model_tag in selected_model_names:
            model_name = available_models[model_tag]
            
            try:
                # Use a spinner for better user feedback during generation
                with st.spinner(f"Generating path with {model_tag}..."):
                    current_model_progress_callback = lambda msg: update_progress(msg, model_tag=model_tag)
                    
                    # Run the agent with Composio hosted MCP
                    result = run_agent_sync(
                        use_youtube=use_youtube,
                        use_drive=use_drive,
                        use_notion=use_notion,
                        user_goal=user_goal,
                        progress_callback=current_model_progress_callback,
                        model_name=model_name
                    )
                
                # Store the result in session state
                st.session_state.model_results[model_tag] = result
                st.success(f"Generated successfully with {model_tag}! 🎉")

            except Exception as e:
                error_message = f"An error occurred with {model_tag}: {str(e)}"
                st.error(error_message)
                st.session_state.model_results[model_tag] = f"**Error:** {str(e)}"

        st.session_state.is_generating = False
        # Force a rerun to make the persistent display block appear immediately
        st.rerun()

# Display stored results on rerun for persistence (This block remains unchanged)
if st.session_state.model_results and not st.session_state.is_generating:
    st.markdown("<div class='section-title'>Your Learning Paths</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    tabs = st.tabs(list(st.session_state.model_results.keys()))
    for tab, (model_tag, result_content) in zip(tabs, st.session_state.model_results.items()):
        with tab:
            st.subheader(f"Learning Path from {model_tag} 🧠")
            if isinstance(result_content, dict):
                text = result_content.get("text", "")
                if text:
                    st.markdown("<div class='response-block'>", unsafe_allow_html=True)
                    st.markdown(text)
                    st.markdown("</div>", unsafe_allow_html=True)
                playlists = result_content.get("playlists", [])
                playlist_links = []
                if playlists:
                    for evt in playlists:
                        pid = evt.get("playlist_id")
                        if pid:
                            url = f"https://www.youtube.com/playlist?list={pid}"
                            playlist_links.append(url)
                if playlist_links:
                    unique_links = list(dict.fromkeys(playlist_links))
                    chips = "".join([
                        f"<a class='playlist-chip' href='{url}' target='_blank' rel='noopener noreferrer'>🎧 Playlist {idx+1}</a>"
                        for idx, url in enumerate(unique_links)
                    ])
                    st.markdown(f"<div class='playlist-stack'>{chips}</div>", unsafe_allow_html=True)
                if playlists:
                    with st.expander("Playlist actions (created/added/failed)"):
                        for p in playlists:
                            action = p.get("action")
                            pid = p.get("playlist_id") or p.get("raw")
                            if action == "created_and_added":
                                st.write(f"Created playlist {pid} and added {len(p.get('added', []))} videos")
                            elif action == "auto_added_to_existing":
                                st.write(f"Auto-added to existing playlist {pid}: added={p.get('added', [])} failed={p.get('failed', [])}")
                            elif action == "added_missing":
                                st.write(f"Added missing videos to {pid}: added={p.get('added', [])} failed={p.get('failed', [])}")
                            elif action == "create_failed":
                                st.error(f"Failed to create playlist for reference {pid}: {p.get('error')}")
                            else:
                                st.write(p)
            else:
                st.markdown("<div class='response-block'>", unsafe_allow_html=True)
                st.markdown(result_content)
                st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
