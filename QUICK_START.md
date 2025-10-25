# 🚀 Quick Start Guide - Composio Hosted MCP

## ✅ What This Uses

The project uses **Composio's hosted MCP server** at `https://mcp.composio.com` - no local installation or Pipedream required!

## Step 1: Add Your API Keys

Edit the `.env` file in this directory and add your API keys:

```env
GOOGLE_API_KEY=your_google_ai_studio_key_here
HF_API_KEY=your_huggingface_key_here
COMPOSIO_API_KEY=your_composio_api_key_here
```

### Where to Get API Keys:

1. **Google AI Studio**: https://makersuite.google.com/app/apikey
   - Free tier available
   - Used for Gemini models

2. **Hugging Face**: https://huggingface.co/settings/tokens
   - Free tier available
   - Used for LLaMA, Mistral models

3. **Composio**: https://app.composio.dev/settings
   - Sign up at https://app.composio.dev/
   - Navigate to Settings → API Keys
   - Copy your API key

## Step 2: Connect Your Apps in Composio

You need to connect at least YouTube. Drive and Notion are optional.

### Use Composio Web Dashboard

1. Go to https://app.composio.dev/apps
2. Click "Add Integration"
3. Search for "YouTube" and connect it
4. Optionally connect Drive and/or Notion
5. Follow the OAuth flow to authorize access

## Step 3: Run the Application

```powershell
streamlit run app.py
```

The app will open in your browser at http://localhost:8501

## Step 4: Test It Out

1. In the sidebar, select an AI model (e.g., "Gemini 2.5 Flash")
2. Choose a secondary tool (None, Drive, or Notion)
3. Enter a learning goal, for example:
   ```
   I want to learn Python basics in 3 days
   ```
4. Click "Generate Learning Path"

## 🎯 What You Should See

The app will:
1. ✅ Connect to Composio's hosted MCP server (https://mcp.composio.com)
2. ✅ Search YouTube for relevant tutorial videos using your connected account
3. ✅ Generate a structured 3-day learning path
4. ✅ Display video titles, links, and descriptions
5. ✅ Optionally create Drive/Notion documents if selected

## 🔧 Troubleshooting

### Problem: "COMPOSIO_API_KEY not found"

**Solution**: Make sure your `.env` file exists and has the COMPOSIO_API_KEY line:
```env
COMPOSIO_API_KEY=your_actual_key_here
```

### Problem: "No tools found" or tools not working

**Solution**: Connect your YouTube account via Composio dashboard:
1. Go to https://app.composio.dev/apps
2. Add YouTube integration
3. Complete the OAuth authorization

### Problem: YouTube search returns errors

**Checklist**:
- [ ] YouTube is connected in Composio dashboard
- [ ] COMPOSIO_API_KEY is correct in `.env`
- [ ] You've completed the OAuth flow for YouTube
- [ ] Try reconnecting YouTube in the dashboard

## 📚 Documentation Files

- **QUICK_START.md** (this file) - Getting started
- **README.md** - Project overview

## 💡 Example Learning Goals

Try these to test the system:

```
I want to learn Python basics in 3 days
I want to learn machine learning in 7 days
I want to learn web development in 10 days
I want to learn data science fundamentals in 5 days
I want to learn Docker and Kubernetes in 14 days
```

## 🎓 How It Works

1. You enter a learning goal
2. The app uses LangChain with MCP adapters
3. Connects to Composio's hosted MCP server (mcp.composio.com)
4. Composio routes requests to YouTube API using your connected account
5. The AI agent searches for relevant videos
6. Creates a structured day-by-day learning path
7. Optionally saves to Drive or Notion if selected

## ⚙️ Architecture

```
Streamlit App 
    ↓ 
LangChain MCP Adapter
    ↓ (streamable_http)
Composio Hosted MCP Server (mcp.composio.com)
    ↓ (with your API key)
YouTube/Drive/Notion APIs
```

**Benefits:**
- ✅ No local MCP server needed
- ✅ No Pipedream workflows required
- ✅ Simple configuration (just API keys)
- ✅ Works on Windows without issues

## ⚙️ Advanced Configuration

### Use Different Models

The sidebar lets you compare multiple models:
- Gemini 2.5 Flash (Google - Fast)
- Claude 3 Sonnet (via HuggingFace)
- Mistral Large (HuggingFace)
- Perplexity (HuggingFace)

### Add More Composio Apps

You can connect more apps via the Composio dashboard and update the code to use them.

## 🆘 Still Having Issues?

1. Check your `.env` file has COMPOSIO_API_KEY
2. Verify YouTube is connected at https://app.composio.dev/apps
3. Check Composio dashboard for connection status
4. Make sure all API keys are valid

## ✅ Checklist

Before running, make sure:

- [ ] `.env` file exists with all three API keys (GOOGLE_API_KEY, HF_API_KEY, COMPOSIO_API_KEY)
- [ ] YouTube is connected in Composio dashboard (https://app.composio.dev/apps)
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] You're in the right directory: `cd C:\Users\dell\Downloads\mcp-learning-path-demo-main\mcp-learning-path-demo-main`

---

**Ready to go!** Run `streamlit run app.py` and start learning! 🚀
