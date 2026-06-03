# Learning Path Generator with Model Context Protocol (MCP)

This project is a Streamlit-based web application that generates personalized learning paths using the Model Context Protocol (MCP). It integrates with various services including YouTube, Google Drive, and Notion to create comprehensive learning experiences.

## Features

- 🎯 Generate personalized learning paths based on your goals
- 🎥 Integration with YouTube for video content
- 📁 Google Drive integration for document storage
- 📝 Notion integration for note-taking and organization
- 🚀 Real-time progress tracking
- 🎨 User-friendly Streamlit interface

## Prerequisites

- Python 3.10+
- Google AI Studio API Key
- **Composio API Key** (for direct MCP integration)
- Optional: Hugging Face API Key if you want to use HF-backed models

If you are on Python 3.14, use the current requirements file as-is. The app now
pins a Pillow version that has Windows wheels for this runtime.

If YouTube connect fails with a 401 from Composio, the problem is usually an
invalid or revoked `COMPOSIO_API_KEY`, not the YouTube account itself.

## Installation

1. Clone the repository:

2. Create and activate a virtual environment:

3. Install the required packages:
```bash
pip install -r requirements.txt
```

If pip tries to build Pillow from source on Windows, upgrade pip first and retry:

```bash
py -3 -m pip install --upgrade pip setuptools wheel
py -3 -m pip install -r requirements.txt
```

## Local Setup

1. Copy `.env.example` to `.env` and fill in at least these values:

```bash
GOOGLE_API_KEY=your_google_ai_studio_key
COMPOSIO_API_KEY=your_composio_api_key
COMPOSIO_YOUTUBE_INTEGRATION_ID=your_youtube_integration_id
COMPOSIO_DRIVE_INTEGRATION_ID=your_drive_integration_id
COMPOSIO_NOTION_INTEGRATION_ID=your_notion_integration_id
APP_URL=http://localhost:8501
```

2. Start the app:

```bash
streamlit run app.py
```

**Get API Keys:**
- Google AI Studio: https://makersuite.google.com/app/apikey
- Hugging Face: https://huggingface.co/settings/tokens
- Composio: https://app.composio.dev/settings

## Composio OAuth Setup

Connect the apps you want to use in Composio and copy the auth config IDs into `.env`:

```bash
composio add youtube           # Required
composio add googledrive       # Optional
composio add notion            # Optional
```

The app uses your own connected accounts through Composio. It stores only the
`connectedAccountId` values locally, not Google/Notion passwords or tokens.
The `COMPOSIO_*_INTEGRATION_ID` values in this repo are used as auth config IDs
for Composio's hosted connect-link flow.

If you are deploying the app, set `APP_URL` to the public URL of the deployed
site so the OAuth redirect can return to the correct place.

## Running the Application

Local run:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` by default.

## Deployment

The app is ready for platforms that run Streamlit apps directly, such as
Streamlit Community Cloud, Render, or Railway.

For deployment:

1. Set environment variables in the host platform, especially `GOOGLE_API_KEY`,
   `COMPOSIO_API_KEY`, all required `COMPOSIO_*_INTEGRATION_ID` values, and
   `APP_URL`.
2. Use the start command:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

3. If your platform does not provide `$PORT`, use its documented Streamlit
   startup command or the default `streamlit run app.py` for Streamlit Cloud.

## Usage

1. Ensure your `.env` file has all required API keys.
2. Make sure Composio integrations are connected.
3. In the Streamlit app sidebar:
   - Select AI model(s) for comparison
   - Choose secondary tool: None, Drive, or Notion
4. Enter your learning goal, for example: "I want to learn python basics in 3 days".
5. Click "Generate Learning Path" to create your personalized learning plan.

**Note**: YouTube integration is always enabled. Drive and Notion are optional.

## OAuth and Email Login

This app does not use password auth. The email field in the login screen is only
the app identity for the local database.

- For local testing, you can set `DEV_USER_EMAIL` in `.env` to auto-login as
   that email.
- For normal use, leave `DEV_USER_EMAIL` blank and sign in with any email you
   want.
- For provider OAuth, users connect their own Google, Drive, and Notion
   accounts through Composio. That is separate from the app login email.

If you are currently blocked by a Google OAuth screen that only allows a test
mail ID, that is a Google Cloud consent-screen setting. Switch the OAuth consent
screen out of testing mode or add the real user as a test user. Composio will
then be able to complete the OAuth flow for other accounts.

## Project Structure

- `app.py` - Main Streamlit application
- `utils.py` - Utility functions and helper methods
- `prompt.py` - Prompt template
- `requirements.txt` - Project dependencies
