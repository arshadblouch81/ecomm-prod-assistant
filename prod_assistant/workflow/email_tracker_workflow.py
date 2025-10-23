from typing import TypedDict, Literal
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt, Command, RetryPolicy
from langchain_core.tools import tool
from dotenv import load_dotenv
from utils.config_loader import load_config
import requests
import structlog
import os
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType
from langchain_core.prompts import ChatPromptTemplate
from retriever.conventional_retreiver import ConversationalRAG
from retriever.retrieval import Retriever  
from datetime import datetime
import json
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import sqlite3



load_dotenv()
log = structlog.get_logger()
configure = load_config()
# 1. LLM
# -------------------
llm = ChatGoogleGenerativeAI( model="gemini-2.0-flash")
# llm = ChatOpenAI(model="gpt-4")
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()
pending_approvals = {}

os.makedirs("checkpoints", exist_ok=True)

conn = sqlite3.connect(database="checkpoints/email_tracker.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# def retrieve_all_threads():
#     all_threads = {}  # Use dict instead of set

#     for checkpoint in checkpointer.list(None):
#         config = checkpoint.config.get("configurable", {})
#         thread_id = config.get("thread_id")

#         if thread_id:
#             all_threads[thread_id] = {
#                 "status": config.get("status"),
#                 "data": config.get("content")
#             }

#     return all_threads  # or list(all_threads.values()) if you want just the data
def get_latest_states_by_thread():
    latest_states = {}

    for  checkpoint in checkpointer.list(None):
        config = checkpoint.config.get("configurable", {})
        thread_id = config.get("thread_id")

        if thread_id:
            # Overwrite with latest checkpoint (assumes list is ordered or you want last seen)
            channel_values = checkpoint.checkpoint.get("channel_values", {})

            latest_states[thread_id] = {
                "state": channel_values,
                "status": config.get("status"),
                "timestamp": config.get("timestamp"),  # optional
                "content": config.get("content")       # optional
            }
            pending_approvals[thread_id] = {
            "status": "pending",
            "data": channel_values
        }

    return latest_states

get_latest_states_by_thread()
# Define the structure for email classification
class EmailClassification(TypedDict):
    intent: Literal["question","issue", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str

class EmailAgentState(TypedDict):
    # Raw email data
    email_content: str
    sender_email: str
    email_id: str

    # Classification result
    classification: EmailClassification | None

    # Raw search/API results
    search_results: list[str] | None  # List of raw document chunks
    customer_history: dict | None  # Raw customer data from CRM

    # Generated content
    draft_response: str | None
    
def read_email(state: EmailAgentState) -> Command:
    """Extract and parse email content"""
    log.info("Extract and parse email content")
    return Command(
        update={
            "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
        },
        # Optional: specify next node if needed
        # goto="classify_intent"
    )

# def read_email(state: EmailAgentState) -> dict:
#     """Extract and parse email content"""
#     # In production, this would connect to your email service
#     log.info("Extract and parse email content")
#     return {
#         "messages": [HumanMessage(content=f"Processing email: {state['email_content']}")]
#     }

def classify_intent(state: EmailAgentState) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """Use LLM to classify email intent and urgency, then route accordingly"""
    log.info("classify email intent")
    # Create structured LLM that returns EmailClassification dict
    structured_llm = llm.with_structured_output(EmailClassification)

    # Format the prompt on-demand, not stored in state
    classification_prompt = f"""
    Analyze this customer email and classify it:

    Email: {state['email_content']}
    From: {state['sender_email']}

    Provide classification including intent, urgency, topic, and summary.
    """

    # Get structured response directly as dict
    classification = structured_llm.invoke(classification_prompt)

    # Determine next node based on classification
    if classification['intent'] == 'billing' or classification['urgency'] == 'critical':
        goto = "human_review"
    elif classification['intent'] in ['question', 'feature','issue']:
        goto = "search_documentation"
    elif classification['intent'] == 'bug' :
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    # Store classification as a single dict in state
    return Command(
        update={"classification": classification},
        goto=goto
    )

def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Search knowledge base for relevant information"""
    log.info("Search knowledge base for relevant information")
    # Build search query from classification
    
    classification = state.get('classification', {})
 
    query = f"{classification.get('intent', '')} {classification.get('topic', '')} {classification.get('summary', '')} {state.get('email_content', '')}"

    try:
        
        # log.info("fetching Documents", str(query))
        print("Searching document")
     
        search_results =  retriever.invoke(query)   
        print("Document retrieved", str(len(search_results)))
        log.info("Document retrieved : "  + str(len(search_results)))
    
    except Exception as e:
        # For recoverable search errors, store error and continue
        search_results = [f"Search temporarily unavailable: {str(e)}"]
        log.error("Search temporarily unavailable", str(e))
    return Command(
        update={"search_results": search_results},  # Store raw results or error
        goto="draft_response"
    )

def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Create or update bug tracking ticket"""
    log.info("Create or update bug tracking ticket")
    # Create ticket in your bug tracking system
    ticket_id = "BUG-12345"  # Would be created via API

    return Command(
        update={
            "search_results": [f"Bug ticket {ticket_id} created"],
            "current_step": "bug_tracked"
        },
        goto="draft_response"
    )
    
def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """Generate response using context and route based on quality"""
    log.info("Generate draft response using context and route based on quality")
    
    try:
            
        classification = state.get('classification', {})

        # Format context from raw state data on-demand
        context_sections = []

        if state.get('search_results'):
            # Format search results for the prompt
            formatted_docs = "\n".join([f"- {doc}" for doc in state['search_results']])
            context_sections.append(f"Relevant documentation:\n{formatted_docs}")

        if state.get('customer_history'):
            # Format customer data for the prompt
            context_sections.append(f"Customer tier: {state['customer_history'].get('tier', 'standard')}")

        # Build the prompt with formatted context
        draft_prompt = f"""
        Draft a response to this customer email:
        {state['email_content']}

        Email intent: {classification.get('intent', 'unknown')}
        Urgency level: {classification.get('urgency', 'medium')}

        {chr(10).join(context_sections)}

        Guidelines:
        - Be professional and helpful
        - Address their specific concern
        - Use the provided documentation when relevant
        -- in regards/sincerely use Adamas Chatbot as name 
        """

        response = llm.invoke(draft_prompt)

        # Determine if human review needed based on urgency and intent
        needs_review = (
            classification.get('urgency') in ['high', 'critical'] or
            classification.get('intent') == 'complex'
        )
        needs_review=True
        # Route to appropriate next node

        log.info("Draft created!!!!!!!!!!")
        goto = "human_review" if needs_review else "send_reply"

        return Command(
            update={"draft_response": response.content},  # Store only the raw response
            goto=goto
        )
    except Exception as e:
        log.error("Error in Human Review", e)

def human_review(state: EmailAgentState) -> Command[Literal["send_reply", END]]:
    """Pause for human review using interrupt and route based on decision"""
            
    classification = state.get('classification', {})

    # interrupt() must come first - any code before it will re-run on resume
    approval_request = {
        "email_id": state['email_id'],
        "original_email": state['email_content'],
        "draft_response": state['draft_response'],
        "urgency": classification.get('urgency'),
        "intent": classification.get('intent'),
        "action": "Please review and approve/edit this response",
        "timestamp": datetime.utcnow().isoformat()
    }

    # Store in pending approvals (for API to notify user)
    pending_approvals[state['email_id']] = {
        "status": "pending",
        "data": approval_request
    }
    log.info("Pause for human review using interrupt and route based on decision")
    # This is the ONLY interrupt() call - it returns human's decision
    human_decision = interrupt(approval_request)
    
    log.info("Human response received")
    # This code runs AFTER approval is received via API
    if human_decision is None:
        human_decision = {"approved": False}

    pending_approvals[state['email_id']]["status"] = "resolved"

    if human_decision.get("approved"):
        return Command(
            update={
                "draft_response": human_decision.get("edited_response", state['draft_response']),
                "approval_status": "approved",
                "approval_timestamp": datetime.utcnow().isoformat()
            },
            goto="send_reply"
        )
    else:
        return Command(
            update={
                "approval_status": "rejected",
                "approval_timestamp": datetime.utcnow().isoformat()
            },
            goto=END
        )
    
        

def format_email(state: EmailAgentState) -> dict:
    """
    Format email for API submission with proper encoding of line breaks and spaces.
    """
    log.info("formatting Email")
    email_content = state["draft_response"].replace("\n", "<br>")
    
    email_msg = {
        "ServiceId": state["email_id"],
        "ToAddress": [
            {
                "Name": "Adamas Client",
                "Address": state['sender_email']
            }
        ],
        "FromAddress": {
            "Name": configure["apiendpoint"]["from_name"],
            "Address": configure["apiendpoint"]["from_email"]
        },
        "CCAddress": [
            {
                "Name": "Arshad Abbas",
                "Address": "arshadblouch@gmail.com"
            }
        ],
        "Subject": "Reply to Email",
        "Content": email_content ,
        "LeaveType": "None",
        "Notes": "Approved by manager",
        "Body": "",
        "Attachments": []
    }

    return email_msg
      
def send_reply(state: EmailAgentState) -> Command:
    """Send the email response via external API with Bearer authentication"""
    log.info("Send the email response via external API with Bearer authentication")
    
    auth_url = configure["apiendpoint"]["base_url"] + "/api/login" 
    base_url = configure["apiendpoint"]["base_url"] + "/api/graph/send-email/"
    token=''
    #get token    
    try:
        jsonBody = {"BrowserName":"Microsoft Edge","Password":"sysmgr","Username":"sysmgr"}
        headers = {"Content-Type": "application/json"}
        response = requests.post(auth_url, json=jsonBody, headers=headers)
        response_data = response.json()
        token=response_data["access_token"]       
    except requests.RequestException as e :
       raise BaseException("Authentication failed to send email", e)
   
    # Format the message payload
    message = format_email(state)

    # Prepare headers with Bearer token
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(base_url, json=message, headers=headers)
        response.raise_for_status()
        log.info(f"Reply sent successfully : {state['email_content'][:100]}")
        print(f"Reply sent successfully: {state['email_content'][:100]}...")
    except requests.RequestException as e:
        print(f"Failed to send reply: {e}")
        log.info("Failed to send reply", {e})
    return {}

def build_email_agent():
    # Create the graph
    log.info("Creating Graph")    
    workflow = StateGraph(EmailAgentState)

    # Add nodes with appropriate error handling
    workflow.add_node("read_email", read_email)
    workflow.add_node("classify_intent", classify_intent)

    # Add retry policy for nodes that might have transient failures
    workflow.add_node(
        "search_documentation",
        search_documentation,
        retry_policy=RetryPolicy(max_attempts=3)
    )
    workflow.add_node("bug_tracking", bug_tracking)
    workflow.add_node("draft_response", draft_response)
    workflow.add_node("human_review", human_review)
    workflow.add_node("send_reply", send_reply)
  

    # Add only the essential edges
    workflow.add_edge(START, "read_email")
    workflow.add_edge("read_email", "classify_intent")   
    workflow.add_edge("send_reply", END)

    # Compile with checkpointer for persistence
    # memory = MemorySaver()
    # memory = AsyncSqliteSaver("checkpoints/email_tracker.db")

    app  = workflow.compile(checkpointer=checkpointer)
    return app

# Initialize agent
agent_app = build_email_agent()

if __name__ =="__main__" :
    # Test with an urgent billing issue
    initial_state = {
        "email_content": "RE: Facing Server Connection Issue with App Version 25.08.001 while accessing documents?",
        "sender_email": "arshadblouch81@gmail.com",
        "email_id": "email_123",
        "messages": []
    }
    import asyncio
    agent_app = (build_email_agent())
    # Run with a thread_id for persistence
    config = {"configurable": {"thread_id": "customer_123"}}
    result = agent_app.invoke(initial_state, config)
    # The graph will pause at human_review
    # print(f"Draft ready for review: {result['draft_response'][:100]}...")

    # When ready, provide human input to resume
    from langgraph.types import Command
    state = agent_app.get_state(config)
    if state.next and "human_review" in state.next: 
        human_response = Command(
            resume={
                "approved": True,
                "edited_response": "It is okay proceed next"
            }
        )

        # Resume execution
        final_result = agent_app.invoke(human_response, config)
    else :
        final_result = result
    print("Email sent successfully!")
    print(f"Message\n {result['draft_response']}")