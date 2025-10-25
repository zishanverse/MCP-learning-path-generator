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
- Hugging Face API Key
- **Composio API Key** (for direct MCP integration)

## Installation

1. Clone the repository:

2. Create and activate a virtual environment:

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration

### 1. Create `.env` file

Copy `.env.example` to `.env` and add your API keys:

```bash
GOOGLE_API_KEY=your_google_ai_studio_key
HF_API_KEY=your_huggingface_key
COMPOSIO_API_KEY=your_composio_api_key
```

**Get API Keys:**
- Google AI Studio: https://makersuite.google.com/app/apikey
- Hugging Face: https://huggingface.co/settings/tokens
- Composio: https://app.composio.dev/settings

### 2. Connect Composio Integrations

Connect the apps you want to use:

```bash
composio add youtube           # Required
composio add googledrive       # Optional
composio add notion            # Optional
```

**See `COMPOSIO_DIRECT_SETUP.md` for detailed setup instructions.**

## Running the Application

To start the application, run:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` by default.

## Usage

1. Ensure your `.env` file has all required API keys
2. Make sure Composio integrations are connected (see Configuration above)
3. In the Streamlit app sidebar:
   - Select AI model(s) for comparison
   - Choose secondary tool: None, Drive, or Notion
4. Enter your learning goal (e.g., "I want to learn python basics in 3 days")
5. Click "Generate Learning Path" to create your personalized learning plan

**Note**: YouTube integration is always enabled. Drive and Notion are optional.

## Project Structure

- `app.py` - Main Streamlit application
- `utils.py` - Utility functions and helper methods
- `prompt.py` - Prompt template
- `requirements.txt` - Project dependencies
