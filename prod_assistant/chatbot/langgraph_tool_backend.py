# backend.py

import asyncio
from langgraph.graph import StateGraph, START, END
from typing import Literal, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
# from workflow.agentic_workflow_with_mcp_websearch import AgenticRAG
from retriever.retrieval import Retriever  
from retriever.conventional_retreiver import ConversationalRAG
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import structlog
import os
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from langchain_core.prompts import ChatPromptTemplate
from utils.config_loader import load_config
from utils.model_loader import ModelLoader


load_dotenv()
    
required_vars = ["GOOGLE_API_KEY", "ASTRA_DB_API_ENDPOINT", "ASTRA_DB_APPLICATION_TOKEN", "ASTRA_DB_KEYSPACE"]

missing_vars = [var for var in required_vars if os.getenv(var) is None]

if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {missing_vars}")

google_api_key = os.getenv("GOOGLE_API_KEY")
db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

print("Environment variables loaded successfully.")
        

log = structlog.get_logger()
# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI()
llm_google = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()
class EmailClassification(TypedDict):
    intent: Literal["question","issue", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str
#-----------------------
def format_docs(docs) -> str:
    """Format retriever docs into readable context."""
    if not docs:
        return ""
    
    formatted_docs = "\n".join([f"- {doc}" for doc in docs])
    
    return formatted_docs

# -------------------
# 2. Tools
# -------------------


# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    """Re write the text to answer questions properly, remove duplicate and ir-relevant text and refine the context"""
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def retreive_data_from_database(state: ChatState) -> str:
    """
    tool to search email history of support staff to find relevant information and answer to the support call and issues.
    
    
    """
    try:
        # log.info("chat_with_agentic_rag called", msg)
        print("Calling Tool chat_with_agentic_rag")
        messages = state["messages"][-1].content
       
        docs =  retriever.invoke(messages)   
        print("Document retrieved", len(docs))
        context = format_docs(docs)
        return {"messages": [context]}  
      
        
    except Exception as e:
        log.error("Error in chat_with_agentic_rag", error=str(e))
        return f"Error in agentic RAG: {str(e)}"
    
# 5. format email node
def format_message_node(state: ChatState) :
    """
    check email history from message of support staff to find relevant information about issues and answer to the support call 
    and issues and formate the reply with given format.   
    
    """
    try:
        print("formatting message")
        messages = state["messages"][-1].content    
       
      
        log.info("formatting email called")
       # Build the prompt with formatted context
        draft_prompt = f"""            
        You are an assistant designed to answer questions using the provided context. Rely only on the retrieved 
        information to form your response. 
         Guidelines:
        - First say thanks for approaching us.
        - Be professional and helpful
        - Address their specific concern
        - address the given problem or question from text below
         {chr(10).join(messages)}.
       
        If there is no enough information to answer then apologize for not having enough information  

       
        """
       
        response = llm_google.invoke( draft_prompt)
        if response :
            return {"messages": [response.content]}
        else :
            return {"message" : [messages]}
        
    except Exception as e:
        log.error("Error in formatting email", error=str(e))
        return f"Error in formatting email: {str(e)}"
    

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
# Build graph
graph = StateGraph(ChatState)

# Add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("database", retreive_data_from_database)
graph.add_node("format_message_node", format_message_node)


# Add edges
graph.add_edge(START, "database")
graph.add_edge("database", "chat_node")
graph.add_edge("chat_node", "format_message_node")
graph.add_edge("format_message_node", END)


#--------------------compile graph-------------------------------
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


async def main_call():
    """Main async execution"""
    try:
        # ✅ Initialize retriever first
      
        state = {
            'messages': [
                HumanMessage(
                    content="RE: URGENT: Critical Server Connection Issue with App Version 25.08.001 [ITR:0015942]? use chat_with_agentic_rag tool get email data"
                )
            ]
        }
        
        log.info("Processing test query")
        
        # Stream the response
        async for event in chatbot.astream(
            state, 
            config={'configurable': {'thread_id': 'default-21'}}
        ):
            print(event)
        
        # break  # Exit after processing
            
    except Exception as e:
        log.error("Error in main execution", error=str(e))
        raise

if __name__ == "__main__":
    state = {
            'messages': [
                HumanMessage(
                    content="RE: URGENT: Critical Server Connection Issue with App Version 25.08.001 [ITR:0015942]? use chat_with_agentic_rag tool get email data"
                )
            ]
        }
    # Add config with thread_id
    config = {
        "configurable": {
            "thread_id": "conversation_11"  # Unique identifier for this conversation
        }
    }

    # Invoke with config
    result = chatbot.invoke(state, config=config)
    print(result["messages"][-1].content)