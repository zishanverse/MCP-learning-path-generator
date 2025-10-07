from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from prompt import user_goal_prompt
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
import os
from langchain_core.tools import BaseTool
from typing import Optional, Tuple, Any, Callable, Dict, List
import asyncio
import json
from pydantic import BaseModel, Field, field_validator

# HIGHLIGHT: Load environment variables from the .env file
load_dotenv()

cfg = RunnableConfig(recursion_limit=100)

#function to dynamically initialize any supported model
def initialize_model(model_name: str) -> Any:
    if "gemini" in model_name:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)

    # Claude replacement → Hugging Face LLaMA
    elif "claude" in model_name or "llama" in model_name:
        hf_key = os.getenv("HF_API_KEY")
        if not hf_key:
            raise ValueError("Hugging Face API key not found. Please set HF_API_KEY in your .env file.")
         # 1. Create the basic endpoint
        llm = HuggingFaceEndpoint(
            # repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            repo_id="NousResearch/Meta-Llama-3-8B-Instruct", 
            # task="text-generation",
            max_new_tokens=1024,
            huggingfacehub_api_token=hf_key
        )
        
        # 2. Wrap it in a ChatHuggingFace object
        return ChatHuggingFace(llm=llm)
    # Mistral via Hugging Face
    elif "mistral" in model_name:
        hf_key = os.getenv("HF_API_KEY")
        if not hf_key:
            raise ValueError("Hugging Face API key not found. Please set HF_API_KEY in your .env file.")
        # 1. Create the basic endpoint
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            # task="text-generation",
            max_new_tokens=1024,
            huggingfacehub_api_token=hf_key
        )
        
        # 2. Wrap it in a ChatHuggingFace object
        return ChatHuggingFace(llm=llm)

    elif "perplexity" in model_name:
        hf_key = os.getenv("HF_API_KEY")
        if not hf_key:
            raise ValueError("Hugging Face API key not found. Please set HF_API_KEY in your .env file.")
            # Using Perplexity's model from Hugging Face
            llm = HuggingFaceEndpoint(
                repo_id="perplexity-ai/llama-3-8b-instruct",  # you can also try other Perplexity models
                task="text-generation",
                max_new_tokens=1024,
                temperature=0.7,
                huggingfacehub_api_token=hf_key
            )

    else:
        raise ValueError(f"Unknown model: {model_name}")

class ComposioAwareTool(BaseTool):
    """Tool wrapper that handles Composio-specific requirements"""
    
    name: str
    description: str
    original_tool: Any
    connected_account_id: Optional[str] = None
    
    class ComposioInput(BaseModel):
        """Input model that handles common Composio parameters"""
        model_config = {"extra": "allow", "str_strip_whitespace": True}
        
        # Common parameters for different tool types
        q: Optional[str] = Field(None, description="Query parameter")
        query: Optional[str] = Field(None, description="Search query")
        toolkit_name: Optional[str] = Field(None, description="Toolkit name")
        connected_account_id: Optional[str] = Field(None, description="Connected account ID")
        params: Optional[Any] = Field(None, description="Additional parameters (can be dict or string)")
        
        @field_validator('params', mode='before')
        @classmethod
        def parse_params(cls, v):
            """Parse params if it's a JSON string"""
            if isinstance(v, str):
                try:
                    import json
                    return json.loads(v)
                except json.JSONDecodeError:
                    return {"value": v}
            return v
        
        def model_dump(self, **kwargs):
            """Override model_dump to include extra fields"""
            data = super().model_dump(**kwargs)
            # Add any extra fields that were set
            for key, value in self.__dict__.items():
                if key not in data and not key.startswith('_') and value is not None:
                    data[key] = value
            return data
    
    def __init__(self, original_tool: Any, connected_account_id: Optional[str] = None, **kwargs):
        super().__init__(
            name=original_tool.name,
            description=self._create_description(original_tool),
            original_tool=original_tool,
            **kwargs
        )
        self.args_schema = self.ComposioInput
        self.connected_account_id = connected_account_id
    
    def _create_description(self, tool):
        """Create a helpful description based on tool name"""
        name = tool.name.lower()
        base_desc = tool.description or f"Execute {tool.name}"
        
        if 'youtube' in name or 'search' in name:
            return f"{base_desc}. Use 'q' parameter for search query."
        elif 'notion' in name:
            return f"{base_desc}. Notion-related operation."
        elif 'composio' in name:
            if 'search' in name:
                return f"{base_desc}. Search for available tools."
            elif 'execute' in name:
                return f"{base_desc}. Execute a specific tool with toolkit_name and params."
        
        return base_desc
    
    def _prepare_composio_params(self, **kwargs):
        """Prepare parameters specifically for Composio tools"""
        tool_name = self.name.lower()
        cleaned_params = {}
        
        # Handle different tool types
        if 'search_tools' in tool_name:
            # For COMPOSIO_SEARCH_TOOLS - search for YouTube tools
            cleaned_params = {
                "query": kwargs.get('q') or kwargs.get('query') or "youtube"
            }
        elif 'execute_tool' in tool_name:
            # For COMPOSIO_EXECUTE_TOOL - need toolkit_name and params
            query = kwargs.get('q') or kwargs.get('query') or ""
            
            # Get connected account ID from multiple sources
            account_id = (
                kwargs.get('connected_account_id') or 
                self.connected_account_id or
                "default"  # Fallback to default
            )
            
            cleaned_params = {
                "toolkit_name": kwargs.get('toolkit_name') or "youtube",
                "connected_account_id": account_id,
                "params": {"q": query} if query else kwargs.get('params', {})
            }
            
            # Debug logging
            print(f"DEBUG: Prepared execute_tool params: {cleaned_params}")
            
        elif 'youtube' in tool_name or 'search' in tool_name:
            # For direct YouTube tools
            query = kwargs.get('q') or kwargs.get('query') or ""
            if query:
                cleaned_params = {"q": query}
        elif 'notion' in tool_name:
            # For Notion tools - pass through relevant params
            for key, value in kwargs.items():
                if value is not None and key not in ['connected_account_id', 'toolkit_name']:
                    cleaned_params[key] = value
        else:
            # Default: pass through non-None values
            for key, value in kwargs.items():
                if value is not None:
                    cleaned_params[key] = value
        
        return cleaned_params
    
    async def _arun(self, **kwargs) -> str:
        """Execute with Composio-aware parameter handling"""
        try:
            # Prepare parameters for this specific tool
            cleaned_params = self._prepare_composio_params(**kwargs)
            
            print(f"Executing {self.name} with params: {cleaned_params}")
            
            # Execute the tool
            result = await self.original_tool.ainvoke(cleaned_params)
            
            #print(f"Raw result from {self.name}: {result}")
            
            # Handle the result
            if isinstance(result, dict):
                # Check if the result indicates an error
                if not result.get('successful', True):
                    error_msg = result.get('error', 'Unknown error')
                    print(f"Tool {self.name} failed: {error_msg}")
                    
                    # Handle specific error cases
                    if 'youtube' in self.name.lower():
                        if 'Missing' in error_msg or 'cannot both be None' in error_msg:
                            return self._fallback_youtube_search(kwargs.get('q', ''))
                    
                    return f"Tool execution failed: {error_msg}"
                
                # Return successful result with proper formatting
                data = result.get('data', result)
                
                # Format based on tool type and data content
                if self._is_youtube_data(result) or self._is_youtube_data(data):
                    return self._format_youtube_results(result)
                elif 'notion' in self.name.lower():
                    return self._format_notion_results(data)
                else:
                    # General formatting for other tools
                    if isinstance(data, dict) and len(str(data)) > 500:
                        return f"✅ {self.name} executed successfully with {len(data)} items returned."
                    else:
                        return json.dumps(data, indent=2) if isinstance(data, dict) else str(data)
            
            return str(result) if result is not None else "Operation completed successfully"
            
        except Exception as e:
            error_msg = f"Error executing {self.name}: {str(e)}"
            print(error_msg)
            
            # Provide fallback for YouTube search
            if 'youtube' in self.name.lower() and kwargs.get('q'):
                return self._fallback_youtube_search(kwargs.get('q', ''))
            
            return error_msg
    
    def _is_youtube_data(self, data):
        """Check if data contains YouTube video information"""
        if not isinstance(data, dict):
            return False
        
        # Check various possible structures for YouTube data
        indicators = ['items', 'videoId', 'youtube', 'response_data']
        data_str = str(data).lower()
        
        return any(indicator in data_str for indicator in indicators) and 'video' in data_str
    
    def _format_youtube_results(self, result_data):
        """Format YouTube API results into readable format with links"""
        try:
            # Handle multiple possible nested structures
            items = None
            
            if isinstance(result_data, dict):
                # Direct items access
                if 'items' in result_data:
                    items = result_data['items']
                # Nested data.response_data.items structure
                elif 'data' in result_data:
                    data_section = result_data['data']
                    if isinstance(data_section, dict):
                        if 'response_data' in data_section:
                            response_data = data_section['response_data']
                            if isinstance(response_data, dict) and 'items' in response_data:
                                items = response_data['items']
                        elif 'items' in data_section:
                            items = data_section['items']
                # Direct response_data.items structure
                elif 'response_data' in result_data:
                    response_data = result_data['response_data']
                    if isinstance(response_data, dict) and 'items' in response_data:
                        items = response_data['items']
            
            if not items or not isinstance(items, list):
                return f"No video results found. Raw data structure: {str(result_data)[:200]}..."
            
            formatted_videos = []
            items = items[:8]  # Limit to top 8 results for better readability
            
            for i, item in enumerate(items, 1):
                try:
                    # Handle different video ID structures
                    video_id = ""
                    if isinstance(item, dict):
                        if 'id' in item:
                            if isinstance(item['id'], dict) and 'videoId' in item['id']:
                                video_id = item['id']['videoId']
                            elif isinstance(item['id'], str):
                                video_id = item['id']
                        elif 'videoId' in item:
                            video_id = item['videoId']
                    
                    # Get snippet information
                    snippet = item.get('snippet', {}) if isinstance(item, dict) else {}
                    
                    title = snippet.get('title', 'No title available')
                    description = snippet.get('description', 'No description available')
                    channel = snippet.get('channelTitle', 'Unknown channel')
                    published = snippet.get('publishedAt', snippet.get('publishTime', ''))
                    if published and len(published) >= 10:
                        published = published[:10]
                    else:
                        published = 'Unknown date'
                    
                    # Create YouTube URLs
                    if video_id:
                        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
                        embed_url = f"https://www.youtube.com/embed/{video_id}"
                    else:
                        youtube_url = "URL not available"
                        embed_url = "Embed not available"
                    
                    # Truncate description to 120 characters for better readability
                    short_desc = (description[:120] + "...") if len(description) > 120 else description
                    
                    video_info = f"""
**Video {i}: {title}**
- **🎬 YouTube URL**: {youtube_url}
- **📺 Embed URL**: {embed_url}
- **📢 Channel**: {channel}
- **📅 Published**: {published}
- **📝 Description**: {short_desc}
---"""
                    formatted_videos.append(video_info)
                    
                except Exception as video_error:
                    print(f"Error processing video {i}: {video_error}")
                    continue
            
            if formatted_videos:
                header = f"🎯 **Found {len(formatted_videos)} YouTube Videos:**\n"
                return header + "\n".join(formatted_videos)
            else:
                return "No videos could be processed from the results."
            
        except Exception as e:
            error_msg = f"Error formatting YouTube results: {str(e)}"
            debug_info = f"\nRaw data preview: {str(result_data)[:300]}..."
            return error_msg + debug_info
    
    def _format_notion_results(self, result_data):
        """Format Notion results into readable format"""
        try:
            if isinstance(result_data, dict):
                if 'url' in result_data:
                    return f"✅ Notion page created successfully!\n**📄 Page URL**: {result_data['url']}"
                elif 'id' in result_data:
                    return f"✅ Notion operation completed successfully!\n**🆔 Resource ID**: {result_data['id']}"
                else:
                    return f"✅ Notion operation completed: {json.dumps(result_data, indent=2)}"
            return str(result_data)
        except Exception as e:
            return f"Error formatting Notion results: {str(e)}"
    
    def _fallback_youtube_search(self, query: str) -> str:
        """Fallback response for YouTube search when tools fail"""
        if not query:
            return "Please provide a search query for YouTube videos."
        
        return f"""🔧 **YouTube Search Configuration Issue**

I attempted to search YouTube for '{query}' but encountered a configuration issue with the tools.

**Troubleshooting Steps:**
1. ✓ Verify your YouTube integration is properly set up in Composio/Pipedream
2. ✓ Ensure the connected account has proper permissions
3. ✓ Check that your connected_account_id is correctly configured
4. ✓ Try searching manually for '{query}' on YouTube

**Suggested YouTube Search Terms:**
- "{query} tutorial"
- "{query} basics"
- "{query} beginner guide"
- "{query} step by step"

**Manual Search URL**: https://www.youtube.com/results?search_query={query.replace(' ', '+')}

For now, I'll continue with the learning path creation using general recommendations."""
    
    def _run(self, **kwargs) -> str:
        """Synchronous execution fallback"""
        try:
            return asyncio.run(self._arun(**kwargs))
        except Exception as e:
            error_msg = f"Error executing {self.name}: {str(e)}"
            print(error_msg)
            return error_msg

async def setup_agent_with_tools(
    # google_api_key: str,
    youtube_pipedream_url: str,
    drive_pipedream_url: Optional[str] = None,
    notion_pipedream_url: Optional[str] = None,
    connected_account_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    model_name: str = "gemini-2.5-flash"
) -> Any:
    """
    Set up the agent with Composio-aware tools
    """
    try:
        if progress_callback:
            progress_callback("Setting up agent with tools... ✅")
        
        # Initialize tools configuration
        tools_config = {
            "youtube": {
                "url": youtube_pipedream_url,
                "transport": "streamable_http"
            }
        }

        if drive_pipedream_url:
            tools_config["drive"] = {
                "url": drive_pipedream_url,
                "transport": "streamable_http"
            }
            if progress_callback:
                progress_callback("Added Google Drive integration... ✅")

        if notion_pipedream_url:
            tools_config["notion"] = {
                "url": notion_pipedream_url,
                "transport": "streamable_http"
            }
            if progress_callback:
                progress_callback("Added Notion integration... ✅")

        if progress_callback:
            progress_callback("Initializing MCP client... ✅")
        
        # Initialize MCP client
        mcp_client = MultiServerMCPClient(tools_config)
        
        if progress_callback:
            progress_callback("Getting and processing tools... ✅")
        
        # Get original tools
        original_tools = await mcp_client.get_tools()
        
        print(f"Found {len(original_tools)} original tools")
        for i, tool in enumerate(original_tools):
            print(f"  {i}: {tool.name}")
        
        # Create Composio-aware wrapped tools
        smart_tools = []
        for i, tool in enumerate(original_tools):
            try:
                # Create Composio-aware wrapper with connected account ID
                smart_tool = ComposioAwareTool(
                    tool, 
                    connected_account_id=connected_account_id
                )
                smart_tools.append(smart_tool)
                print(f"✅ Wrapped tool: {tool.name}")
                
            except Exception as e:
                print(f"❌ Failed to wrap tool {tool.name}: {e}")
                continue
        
        if not smart_tools:
            raise Exception("No tools could be successfully wrapped")
        
        if progress_callback:
            progress_callback(f"Successfully processed {len(smart_tools)} tools... ✅")
        
        # ✅ HIGHLIGHT: Use free initialize_model
        model = initialize_model(model_name)
        
        if progress_callback:
            progress_callback("Creating agent with smart tools... ✅")
        
        # Create agent
        agent = create_react_agent(model, smart_tools)
        
        if progress_callback:
            progress_callback("Agent setup complete! ✅")
        
        return agent
        
    except Exception as e:
        print(f"Error in setup_agent_with_tools: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def run_agent_sync(
    # google_api_key: str,
    youtube_pipedream_url: str,
    drive_pipedream_url: Optional[str] = None,
    notion_pipedream_url: Optional[str] = None,
    user_goal: str = "",
    connected_account_id: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    model_name: str = "gemini-2.5-flash"
) -> str: # 🌟
    """
    Synchronous wrapper for running the agent.
    """
    async def _run():
        try:
            agent = await setup_agent_with_tools(
                # google_api_key=google_api_key,
                youtube_pipedream_url=youtube_pipedream_url,
                drive_pipedream_url=drive_pipedream_url,
                notion_pipedream_url=notion_pipedream_url,
                connected_account_id=connected_account_id,
                progress_callback=progress_callback,
                model_name=model_name
            )
            
            # Enhanced prompt with specific instructions for better output formatting
            enhanced_prompt = f"""User Goal: {user_goal}

{user_goal_prompt}

IMPORTANT TOOL USAGE AND OUTPUT INSTRUCTIONS:
1. When searching for YouTube videos, use the 'q' parameter with your search query
2. Always format your final response as a clean, structured learning path
3. For each day, provide:
    - Day number and topic
    - Specific YouTube video with title, URL, and brief description
    - Key learning objectives for that day
4. Do NOT include raw API responses in your final answer
5. Focus on creating a professional, readable learning plan
6. If tools return errors, provide alternative suggestions but still create a complete plan
7. Use the connected_account_id if required for tool execution

FORMAT EXAMPLE:
# 10-Day AI/ML Learning Path for Startup Business and Freelancing

## Day 1: Introduction to AI and ML
**Topic**: Understanding AI/ML fundamentals for business
**YouTube Video**: [Video Title] - https://www.youtube.com/watch?v=VIDEO_ID
**Description**: Brief explanation of what this video covers
**Learning Objectives**: 
- Understand basic AI/ML concepts
- Learn business applications

Continue this format for all 10 days...

Remember: Create a professional, complete learning path regardless of any tool issues."""
            
            if progress_callback:
                progress_callback("Generating your learning path...")
            
            # Run the agent
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=enhanced_prompt)]},
                config=cfg
            )
            
            if progress_callback:
                progress_callback("Learning path generation complete!")
            
            # Extract the final content from the agent's response
            if isinstance(result, dict) and "messages" in result:
                # The final answer is typically in the content of the last message
                final_message = result["messages"][-1]
                if hasattr(final_message, "content"):
                    return final_message.content
                else:
                    return str(result) # Fallback if content not found
            else:
                return str(result) # Fallback if structure is unexpected
            
        except Exception as e:
            print(f"Error in _run: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # Run in new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_run())
    finally:
        loop.close()