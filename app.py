import streamlit as st
from utils import run_agent_sync
# HIGHLIGHT: Imports from dotenv and os removed, moved to utils.py

st.set_page_config(page_title="MCP POC", page_icon="🤖", layout="wide")

st.title("Model Context Protocol(MCP) - Learning Path Generator")

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


# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Configuration")

# Define the available models and their API names
available_models = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    # "Gemini 1.5 Pro": "gemini-1.5-pro-latest",
    "Claude 3 Sonnet": "claude-3-sonnet-20240229",
    "Mistral Large": "mistral-large-latest",
}

st.sidebar.subheader("AI Model Comparison")
# Allow the user to select multiple models for comparison
selected_model_names = st.sidebar.multiselect(
    "Select AI Models for Comparison:",
    options=list(available_models.keys()),
    default=["Gemini 2.5 Flash", "Mistral Large"]
)
selected_models = [available_models[name] for name in selected_model_names]

# HIGHLIGHT: Removed manual API key input

# pipedream URLs
st.sidebar.subheader("pipedream URLs")
youtube_pipedream_url = st.sidebar.text_input("YouTube URL (Required)", 
    placeholder="Enter your pipedream YouTube URL")

# Secondary tool selection
secondary_tool = st.sidebar.radio(
    "Select Secondary Tool:",
    ["Drive", "Notion"]
)

# Secondary tool URL input
if secondary_tool == "Drive":
    drive_pipedream_url = st.sidebar.text_input("Drive URL", 
        placeholder="Enter your pipedream Drive URL")
    notion_pipedream_url = None
else:
    notion_pipedream_url = st.sidebar.text_input("Notion URL", 
        placeholder="Enter your pipedream Notion URL")
    drive_pipedream_url = None

# Quick guide before goal input
st.info("""
**Quick Guide:**
1. Your API keys are loaded securely from your local **.env** file.
2. Select one or more AI Models for side-by-side comparison.
3. Enter a clear learning goal, for example:
    - "I want to learn python basics in 3 days"
    - "I want to learn data science basics in 10 days"
""")

# Main content area
st.header("Enter Your Goal")
user_goal = st.text_input("Enter your learning goal:",
    help="Describe what you want to learn, and we'll generate a structured path using YouTube content and your selected tool.")

# Progress area
progress_container = st.container()
progress_bar = st.empty()

# HIGHLIGHT: Update progress function to show which model is running
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
    # HIGHLIGHT: Removed Google API key validation
    if not selected_models:
        st.error("Please select at least one AI model for comparison.")
    elif not youtube_pipedream_url:
        st.error("YouTube URL is required. Please enter your pipedream YouTube URL in the sidebar.")
    elif (secondary_tool == "Drive" and not drive_pipedream_url) or (secondary_tool == "Notion" and not notion_pipedream_url):
        st.error(f"Please enter your pipedream {secondary_tool} URL in the sidebar.")
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
                    
                    # Run the agent and get the result
                    result = run_agent_sync(
                        youtube_pipedream_url=youtube_pipedream_url,
                        drive_pipedream_url=drive_pipedream_url,
                        notion_pipedream_url=notion_pipedream_url,
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
    st.header("Your Generated Learning Paths (Comparison)")
    
    model_count = len(st.session_state.model_results)
    if model_count > 0:
        results_columns_display = st.columns(model_count)
        for idx, (model_tag, result_content) in enumerate(st.session_state.model_results.items()):
            with results_columns_display[idx]:
                st.subheader(f"Learning Path from {model_tag} 🧠")
                st.markdown(result_content)
