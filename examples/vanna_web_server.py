"""
Vanna AI Web Server with Chat Interface

This script sets up a FastAPI web server with Vanna's built-in chat UI.
Access the chat interface at http://localhost:8000

Usage:
    python vanna_web_server.py
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()


# Simple user resolver for demo
class SimpleUserResolver:
    """Simple user resolver that always returns a demo user."""

    async def resolve_user(self, request_context):
        """Resolve to a demo user."""
        from vanna import User

        return User(id="demo-user", username="demo", email="demo@example.com")


# Custom System Prompt Builder with Schema Injection
class SchemaAwareSystemPromptBuilder:
    """System prompt builder that includes database schema information."""

    def __init__(self, schema_info: str, base_prompt: str = None):
        """Initialize with schema information.

        Args:
            schema_info: String describing the database schema
            base_prompt: Optional custom base prompt
        """
        self.schema_info = schema_info
        self.base_prompt = base_prompt

    async def build_system_prompt(self, user, tools):
        """Build system prompt with schema context."""
        from datetime import datetime

        tool_names = [tool.name for tool in tools]
        today_date = datetime.now().strftime("%Y-%m-%d")

        prompt_parts = [
            f"You are Vanna, an AI data analyst assistant. Today's date is {today_date}.",
            "",
            "=== DATABASE SCHEMA ===",
            self.schema_info,
            "",
            "=== IMPORTANT INSTRUCTIONS ===",
            "- You can ONLY query the table(s) listed above. Do not attempt to query other tables.",
            "- Use the EXACT column names as shown in the schema above.",
            "- If a user asks about a concept (like 'category'), match it to the closest column name in the schema.",
            "- Always verify column names against the schema before writing SQL.",
            "",
            "=== RESPONSE GUIDELINES ===",
            "- When you execute a query, the raw result is shown to the user outside your response.",
            "- Focus on summarizing and interpreting results, not repeating raw data.",
            "- Use the available tools to help the user accomplish their goals.",
            "",
        ]

        if tools:
            prompt_parts.append(f"Available tools: {', '.join(tool_names)}")

        return "\n".join(prompt_parts)


def fetch_schema_sync(host, port, database, user, password):
    """Fetch the v_bike_events schema synchronously at startup."""
    import psycopg2

    schema_query = """
    SELECT 
        column_name,
        data_type,
        is_nullable
    FROM information_schema.columns
    WHERE table_schema = 'public' 
      AND table_name = 'v_bike_events'
    ORDER BY ordinal_position;
    """

    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            sslmode="require",
        )
        cur = conn.cursor()
        cur.execute(schema_query)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return "Table 'public.v_bike_events' not found or has no columns."

        schema_lines = ["Table: public.v_bike_events", "Columns:"]
        for row in rows:
            col_name, data_type, nullable = row
            null_str = "" if nullable == "YES" else " (NOT NULL)"
            schema_lines.append(f"  - {col_name}: {data_type}{null_str}")

        return "\n".join(schema_lines)

    except Exception as e:
        return f"Could not fetch schema: {e}"


def validate_env_vars():
    """Validate that all required environment variables are set."""
    # Get the selected LLM provider
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    # Common required vars
    required_vars = {
        "SUPABASE_HOST": "Supabase Host",
        "SUPABASE_DATABASE": "Supabase Database Name",
        "SUPABASE_USER": "Supabase User",
        "SUPABASE_PASSWORD": "Supabase Password",
    }

    # Provider-specific required vars
    if llm_provider == "gemini":
        required_vars["GOOGLE_API_KEY"] = "Google API Key (for Gemini)"
    elif llm_provider == "openai":
        required_vars["OPENAI_API_KEY"] = "OpenAI API Key"
    else:
        required_vars["ANTHROPIC_API_KEY"] = "Anthropic API Key"

    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value.startswith("your_"):
            missing_vars.append(f"  - {var} ({description})")

    if missing_vars:
        print("Error: Missing or incomplete environment variables in .env file:")
        print("\n".join(missing_vars))
        print("\nPlease edit the .env file and add your credentials.")
        sys.exit(1)

    print(f"All required environment variables are set (using {llm_provider.upper()})")


def create_app():
    """Create and configure the FastAPI application."""

    # Validate environment
    validate_env_vars()

    # Import Vanna components
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
        from vanna import Agent, AgentConfig
        from vanna.core.registry import ToolRegistry
        from vanna.tools import RunSqlTool, VisualizeDataTool, LocalFileSystem
        from vanna.integrations.local.agent_memory import DemoAgentMemory
        from vanna.servers.fastapi.routes import register_chat_routes
        from vanna.servers.base import ChatHandler

        # Import our custom restricted SQL runner
        from restricted_sql_runner import RestrictedPostgresRunner
    except ImportError as e:
        print(f"Error importing components: {e}")
        print(
            "Make sure you have activated the virtual environment and installed dependencies."
        )
        sys.exit(1)

    # Get LLM provider configuration
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    # Initialize LLM service based on provider
    if llm_provider == "gemini":
        try:
            from vanna.integrations.google import GeminiLlmService
        except ImportError as e:
            print(f"Error importing Gemini: {e}")
            print(
                "Install with: pip install 'vanna[gemini]' or pip install google-genai"
            )
            sys.exit(1)

        google_api_key = os.getenv("GOOGLE_API_KEY")
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")

        print(f"Using Gemini Model: {gemini_model}")
        llm = GeminiLlmService(model=gemini_model, api_key=google_api_key)
    elif llm_provider == "openai":
        try:
            from vanna.integrations.openai import OpenAILlmService
        except ImportError as e:
            print(f"Error importing OpenAI: {e}")
            print("Install with: pip install 'vanna[openai]' or pip install openai")
            sys.exit(1)

        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        print(f"Using OpenAI Model: {openai_model}")
        llm = OpenAILlmService(model=openai_model, api_key=openai_api_key)
    else:
        from vanna.integrations.anthropic import AnthropicLlmService

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")

        print(f"Using Anthropic Model: {anthropic_model}")
        llm = AnthropicLlmService(model=anthropic_model, api_key=anthropic_api_key)

    supabase_host = os.getenv("SUPABASE_HOST")
    supabase_port = int(os.getenv("SUPABASE_PORT", "5432"))
    supabase_database = os.getenv("SUPABASE_DATABASE")
    supabase_user = os.getenv("SUPABASE_USER")
    supabase_password = os.getenv("SUPABASE_PASSWORD")

    print(
        f"Connecting to Supabase: {supabase_host}:{supabase_port}/{supabase_database}"
    )
    print("Restricting queries to: public.v_bike_events view only")

    # Fetch database schema at startup for context injection
    print("Fetching database schema...")
    schema_info = fetch_schema_sync(
        host=supabase_host,
        port=supabase_port,
        database=supabase_database,
        user=supabase_user,
        password=supabase_password,
    )
    print(f"Schema loaded:\n{schema_info}")

    # Create schema-aware system prompt builder
    system_prompt_builder = SchemaAwareSystemPromptBuilder(schema_info=schema_info)

    # Initialize Restricted PostgreSQL Runner for Supabase
    postgres_runner = RestrictedPostgresRunner(
        allowed_tables=["public.v_bike_events"],
        host=supabase_host,
        port=supabase_port,
        database=supabase_database,
        user=supabase_user,
        password=supabase_password,
        sslmode="require",
    )

    # Create shared FileSystem for storing charts and data
    file_system = LocalFileSystem(working_directory="./vanna_data")

    # Create SQL tool with file system
    sql_tool = RunSqlTool(sql_runner=postgres_runner, file_system=file_system)

    # Create tool registry and register tools
    tool_registry = ToolRegistry()
    tool_registry.register_local_tool(sql_tool, access_groups=[])

    # Register visualization tool for chart generation
    try:
        viz_tool = VisualizeDataTool(file_system=file_system)
        tool_registry.register_local_tool(viz_tool, access_groups=[])
        print("✓ Visualization tool enabled")
    except ImportError as e:
        print(f"⚠️  Visualization tool disabled: {e}")
        print("   Charts won't be generated. This is optional.")

    # Create user resolver and agent memory
    user_resolver = SimpleUserResolver()
    agent_memory = DemoAgentMemory()

    # Configure UI features to show SQL queries to all users
    from vanna.core.agent.config import UiFeatures, UiFeature

    ui_features = UiFeatures(
        feature_group_access={
            # Allow all users to see these features (empty list = accessible to all)
            UiFeature.UI_FEATURE_SHOW_TOOL_NAMES: [],  # Show tool names (e.g., "run_sql")
            UiFeature.UI_FEATURE_SHOW_TOOL_ARGUMENTS: [],  # Show SQL query text
            UiFeature.UI_FEATURE_SHOW_TOOL_ERROR: [],  # Show error messages
            UiFeature.UI_FEATURE_SHOW_TOOL_INVOCATION_MESSAGE_IN_CHAT: [],  # Show in chat
            UiFeature.UI_FEATURE_SHOW_MEMORY_DETAILED_RESULTS: [],  # Show memory results
        }
    )

    # Create the agent with schema-aware system prompt and SQL visibility enabled
    agent = Agent(
        llm_service=llm,
        tool_registry=tool_registry,
        user_resolver=user_resolver,
        agent_memory=agent_memory,
        system_prompt_builder=system_prompt_builder,
        config=AgentConfig(
            stream_responses=True,
            ui_features=ui_features,
        ),
    )

    # Create FastAPI app
    app = FastAPI(
        title="Vanna AI Events Explorer",
        description="Natural language interface to query the public.v_bike_events view",
        version="1.0.0",
    )

    # Register Vanna chat routes
    chat_handler = ChatHandler(agent)
    register_chat_routes(app, chat_handler)

    # Create home page with chat interface
    @app.get("/", response_class=HTMLResponse)
    async def home():
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Vanna AI - Events Explorer</title>
            <script src="https://img.vanna.ai/vanna-components.js"></script>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                }
                
                header {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    padding: 1.5rem 2rem;
                    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
                }
                
                h1 {
                    color: #2d3748;
                    font-size: 1.75rem;
                    font-weight: 700;
                }
                
                .subtitle {
                    color: #718096;
                    font-size: 0.95rem;
                    margin-top: 0.5rem;
                }
                
                .info-banner {
                    background: rgba(255, 255, 255, 0.15);
                    backdrop-filter: blur(10px);
                    color: white;
                    padding: 1rem 2rem;
                    margin: 1rem 2rem;
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.3);
                }
                
                .info-banner h3 {
                    font-size: 1rem;
                    margin-bottom: 0.5rem;
                    font-weight: 600;
                }
                
                .info-banner ul {
                    list-style: none;
                    padding-left: 1.5rem;
                }
                
                .info-banner li {
                    margin: 0.25rem 0;
                    position: relative;
                }
                
                .info-banner li:before {
                    content: "→";
                    position: absolute;
                    left: -1.5rem;
                }
                
                .chat-container {
                    flex: 1;
                    display: flex;
                    margin: 1rem 2rem 2rem;
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    overflow: hidden;
                }
                
                vanna-chat {
                    width: 100%;
                    height: 100%;
                    min-height: 600px;
                }
                
                .security-note {
                    background: rgba(255, 193, 7, 0.2);
                    border-left: 4px solid #ffc107;
                    padding: 0.75rem;
                    margin-top: 0.75rem;
                    border-radius: 4px;
                    font-size: 0.875rem;
                }
                
                @media (max-width: 768px) {
                    header, .info-banner, .chat-container {
                        margin: 0.5rem;
                        padding: 1rem;
                    }
                    
                    h1 {
                        font-size: 1.5rem;
                    }
                }
            </style>
        </head>
        <body>
            <header>
                <h1>🤖 Vanna AI Events Explorer</h1>
                <p class="subtitle">Ask questions about your events in natural language</p>
            </header>
            
            <div class="info-banner">
                <h3>💡 Try asking questions like:</h3>
                <ul>
                    <li>How many events are in the database?</li>
                    <li>Show me the top 10 cities by event count</li>
                    <li>What are the most common event categories?</li>
                    <li>How many bike-related events are there?</li>
                    <li>Show me events from the last month</li>
                </ul>
                <div class="security-note">
                    🔒 <strong>Security:</strong> Queries are restricted to the public.v_bike_events view only
                </div>
            </div>
            
            <div class="chat-container">
                <vanna-chat 
                    sse-endpoint="/api/vanna/v2/chat_sse"
                    theme="light">
                </vanna-chat>
            </div>
        </body>
        </html>
        """

    print("\n" + "=" * 60)
    print("✓ Vanna AI Web Server is ready!")
    print("=" * 60)
    print("\n🌐 Open your browser and go to:")
    print("   http://localhost:8000")
    print("\n" + "=" * 60 + "\n")

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Get host and port from environment or use defaults
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8000"))

    # Run the server
    uvicorn.run(app, host=host, port=port)
