# backend.py
import asyncio
import os
from pathlib import Path
import httpx
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from retriever.retrieval import Retriever
from utils.config_loader import load_config
from utils.model_loader import ModelLoader
import sqlite3
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import structlog
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, MessagesState
import structlog


load_dotenv()

log = structlog.get_logger()

# Global variables
retriever = None
chatbot = None
checkpointer = None

google_api_key = os.getenv("GOOGLE_API_KEY")
db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")
open_api_key = os.getenv("OPENAI_API_KEY")  # ✅ Fixed typo: OPEN_API_KEY -> OPENAI_API_KEY
        
# -------------------
# 1. LLM
# -------------------
model_loader = ModelLoader()
config = load_config()
llm = model_loader.load_llm()

# ✅ Load retriever asynchronously properly
retriever_obj = Retriever()
retriever = None  # Will be initialized async

async def initialize_retriever():
    """Initialize retriever asynchronously"""
    global retriever
    if retriever is None:
        retriever =  retriever_obj.load_retriever()
        log.info("Retriever initialized successfully")
    return retriever

# ---------- Helpers ----------
def format_docs(docs) -> str:
    """Format retriever docs into readable context."""
    if not docs:
        return ""
    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)
    return "\n\n---\n\n".join(formatted_chunks)

# -------------------
# 2. Tools
# -------------------

@tool
async def get_traccs_email_info(query: str) -> str:
    """
    This is tool to access data of email about Traccs product which is a community care 
    and time recording system. It provides service information, roster information, 
    Attendance Time Recording, Payroll, Recipient Data, Staff and carer Information.
    Retrieve email information of product Traccs email a community care and time recording 
    system emails conservation of support staff with clients. Search for a given query from 
    local retriever extract answer of query and reply with formal email to the question asked 
    in email. At the end of message add a thank you note and signature with name as Traccs chatbot.
    """
    try:
        log.info("get_traccs_email_info called", query=query)
        
        # ✅ Ensure retriever is initialized
        current_retriever = await initialize_retriever()
        
        if not current_retriever:
            return "Retriever not initialized."
        
        # ✅ Use ainvoke for async retrieval
        docs = await current_retriever.ainvoke(query)
        context = format_docs(docs)
        
        if not context.strip():
            return "No local results found."
        
        # ✅ Format as email
        email_response = f"""
Based on the Traccs system information:

{context}

Thank you for your inquiry regarding the Traccs community care and time recording system.

Best regards,
Traccs Chatbot
"""
        return email_response
        
    except Exception as e:
        import traceback
        log.error("Error in get_traccs_email_info", error=str(e), traceback=traceback.format_exc())
        return f"Error retrieving product info: {str(e)}"

@tool
async def get_product_info(query: str) -> str:
    """
    Retrieve product information for a given query from local retriever.
    Use this for questions about product features, pricing, ratings, and reviews.
    """
    try:
        log.info("get_product_info called", query=query)
        
        # ✅ Ensure retriever is initialized
        current_retriever = await initialize_retriever()
        
        if not current_retriever:
            return "Retriever not initialized."
        
        # ✅ Use ainvoke for async retrieval
        docs = await current_retriever.ainvoke(query)
        context = format_docs(docs)
        
        if not context.strip():
            return "No local results found."
        
        return context
        
    except Exception as e:
        import traceback
        log.error("Error in get_product_info", error=str(e), traceback=traceback.format_exc())
        return f"Error retrieving product info: {str(e)}"

@tool
async def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo if retriever has no results.
    Use this for current events, news, or information not in the local database.
    """
    try:
        log.info("web_search called", query=query)
        
        # ✅ Run synchronous DuckDuckGo in executor
        loop = asyncio.get_event_loop()
        duckduckgo = DuckDuckGoSearchRun(region="us-en")
        result = await loop.run_in_executor(None, duckduckgo.run, query)
        
        return result
        
    except Exception as e:
        log.error("Error in web_search", error=str(e))
        return f"Error during web search: {str(e)}"

@tool
async def chat_with_agentic_rag(msg: str) -> str:
    """
    Chat tool using Agentic RAG workflow with MCP web search and FAISS retriever.
    Use this for complex queries that require multi-step reasoning and web search.
    """
    try:
        log.info("chat_with_agentic_rag called", msg=msg)
        
        # Import here to avoid circular imports
        from workflow.agentic_workflow_with_mcp_websearch import AgenticRAG
        
        rag_agent = AgenticRAG()
        answer = await rag_agent.run(msg)
        
        log.info("Agentic RAG response generated", answer_length=len(answer))
        return answer
        
    except Exception as e:
        log.error("Error in chat_with_agentic_rag", error=str(e))
        return f"Error in agentic RAG: {str(e)}"

# ✅ Define tools list - all are now proper @tool decorated functions
tools = [get_traccs_email_info, get_product_info, web_search, chat_with_agentic_rag]

# ✅ Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

# -------------------
# 5. Checkpointer
# -------------------

async def create_checkpointer():
    # conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)   
    # checkpointer = AsyncSqliteSaver(conn=conn)
    
    # ✅ CORRECT - Pass the database path as a string
    checkpointer = SqliteSaver.from_conn_string("chatbot.db")
    
    return checkpointer


# -------------------
# 6. Graph
# -------------------
async def create_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge('tools', 'chat_node')
    graph.add_edge('chat_node', END)
    # checkpointer = await create_checkpointer()
   
    # chatbot = graph.compile(checkpointer=checkpointer)
    return graph

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    """Retrieve all conversation thread IDs from checkpointer"""
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
    except Exception as e:
        log.error("Error retrieving threads", error=str(e))
    return list(all_threads)

# -------------------
# 8. Main execution
# -------------------


async def initialize_chatbot():
    """Initialize the chatbot graph with checkpointer"""
    
    
    # ✅ Create checkpointer using async with
    async with AsyncSqliteSaver.from_conn_string("chatbot.db") as cp:
        graph = await create_graph()
        # Compile with checkpointer
        chatbot = graph.compile(checkpointer=checkpointer)
        
        log.info("Chatbot initialized successfully")
        
        # Keep the context alive by yielding control back
        return chatbot



async def async_initialize():
    """
    Async initialization wrapper that handles both async and sync initialization.
    
    This method:
    1. Initializes async components (retriever, embeddings, etc.)
    2. Initializes sync components (chatbot graph with checkpointer)
    3. Returns the compiled chatbot ready for use
    
    Returns:
        Compiled chatbot graph with checkpointer
    """
    try:
        log.info("🚀 Starting async initialization...")
        
        # Step 1: Initialize async components
        await initialize_retriever()
        
        # Step 2: Initialize sync components
        bot = create_graph()
        
        log.info("✅ Async initialization completed successfully")
        return bot
        
    except Exception as e:
        log.error("❌ Failed during async initialization", error=str(e))
        raise
# chatbot = asyncio.run(create_graph())
# ✅ Initialize at module level
try:
    # If you need async initialization, use this:
    chatbot = asyncio.run(async_initialize())
    
    # For synchronous initialization:
    # chatbot = initialize_chatbot()
    log.info("✅ Module initialized successfully")
    
except Exception as e:
    log.error("Failed to initialize", error=str(e))
    chatbot = None


async def main():
    """Main async execution"""
    try:
        # ✅ Initialize retriever first
        await initialize_retriever()
        log.info("Backend initialized successfully")
        
        # ✅ Initialize chatbot with proper async context
        bot = await create_graph()
        bot.compile(checkpointer=checkpointer)
        state = {
            'messages': [
                HumanMessage(
                    content="RE: URGENT: Critical Server Connection Issue with App Version 25.08.001 [ITR:0015942]?"
                )
            ]
        }
        
        log.info("Processing test query")
        
        # Stream the response
        async for event in bot.astream(
            state, 
            config={'configurable': {'thread_id': 'default'}}
        ):
            print(event)
        
        # break  # Exit after processing
            
    except Exception as e:
        log.error("Error in main execution", error=str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())