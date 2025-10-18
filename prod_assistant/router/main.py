
# import uvicorn
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from langchain_core.messages import HumanMessage
# from  workflow.agentic_workflow_with_mcp_websearch import AgenticRAG

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------- FastAPI Endpoints ----------
# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("chat.html", {"request": request})


# @app.post("/get", response_class=HTMLResponse)
# async def chat(msg: str = Form(...)):
#     """Call the Agentic RAG workflow."""
#     rag_agent = AgenticRAG()
#     answer = rag_agent.run(msg)   # run() already returns final answer string
#     print(f"Agentic Response: {answer}")
#     return answer
# requirements.txt additions:
# pyodbc
# python-jose[cryptography]
# passlib[bcrypt]
# python-multipart

import uvicorn
from fastapi import FastAPI, Header, Query, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional, Dict
import pyodbc
import hashlib
import bcrypt
from passlib.context import CryptContext
from jose import JWTError, jwt
from utils.config_loader import load_config
from fastapi.responses import RedirectResponse
import secrets
from  model.models import ApprovalResponse, EmailRequest
from fastapi import WebSocket, WebSocketDisconnect
from workflow.email_tracker_workflow import agent_app, pending_approvals, EmailAgentState
import asyncio
from langgraph.types import Command


#print(pyodbc.drivers())

config=load_config()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Example of hashing a password safely
raw_password = "secure123"  # or any user input

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active tokens temporarily (use Redis in production)
active_sessions = {}

# ---------- Database Connection ----------
def get_db_connection():
    """Create and return SQL Server connection"""
    try:
        conn_str = (
            config["sql_server"]["connection_string"]
        )
        conn = pyodbc.connect(conn_str)
        return conn
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Database connection failed"
        )


# ---------- Password & Token Functions ----------


def hash_password(password: str) -> str:
    # Pre-hash with SHA-256 to handle any length
    prehashed = hashlib.sha256(password.encode('utf-8')).hexdigest()
    # Then bcrypt the hash
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(prehashed.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    # Pre-hash the input password the same way
    prehashed = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return bcrypt.checkpw(prehashed.encode('utf-8'), hashed.encode('utf-8'))

# def verify_password(plain_password, hashed_password):
#     """Verify a plain password against hashed password"""
#     return  pwd_context.verify(plain_password, hashed_password)


# def hash_password(password: str) -> str:
#     return pwd_context.hash(password.encode()[:72])


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config["security"]["SECRET_KEY"], algorithm=config["security"]["ALGORITHM"])
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token from Authorization header"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, config["security"]["SECRET_KEY"], algorithms=[config["security"]["ALGORITHM"]])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )


# ---------- Database Functions ----------
def authenticate_user(username: str, password: str):
    """Authenticate user against SQL Server database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    hashed = hash_password(raw_password)
    print(hashed)

    try:
        # Query user table
        query = "SELECT name as username, password as password_hash, userid as user_id, email, role FROM [User] WHERE name = ?"
        cursor.execute(query, (username,))
        user = cursor.fetchone()
        
        if not user:
            return None
        
        # Verify password
        if not verify_password(password, user.password_hash):
            return None
        
        return {
            "username": user.username,
            "user_id": user.user_id,
            "email": user.email
        }
    except Exception as e:
        print(f"Authentication error: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()


# ---------- FastAPI Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Login page"""
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/main", response_class=HTMLResponse)
async def main(request: Request):
    form = await request.form()
    access_token = form.get("access_token")

    response = RedirectResponse(url="/main-page", status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    

    return response

# Separate GET endpoint for the actual page
@app.get("/main-page", response_class=HTMLResponse)
async def main_page(request: Request):
    token = request.cookies.get("access_token")
    return templates.TemplateResponse("index.html", {"request": request, "token": token})


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Login endpoint - returns JWT token"""
    user = authenticate_user(username, password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=config['security']['ACCESS_TOKEN_EXPIRE_MINUTES'])
    access_token = create_access_token(
        data={
            "sub": user["username"],
            "user_id": user["user_id"],
            "email": user["email"]
        },
        expires_delta=access_token_expires
    )
    
    
    # Store active session
    session_token = secrets.token_urlsafe(32)
    active_sessions[session_token] = access_token
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }


@app.post("/get", response_class=HTMLResponse)
async def chat(
    msg: str = Form(...),
    token_data: dict = Depends(verify_token)   
):
    """Protected chat endpoint - requires valid JWT token"""
    # Token is verified, user is authenticated
    print(f"Token Data: {token_data}")
    username = token_data.get("sub")
    print(f"Authenticated user: {username}")
    
    # Import your workflow here to avoid circular imports
    from workflow.agentic_workflow_with_mcp_websearch import AgenticRAG
    
    rag_agent = AgenticRAG()
    answer = await rag_agent.run(msg)
    print(f"Agentic Response: {answer}")
    return answer

@app.post("/api/chat-message/", response_class=HTMLResponse)
async def chat_api_message(    
    request: Request
):
    """Protected chat endpoint - requires valid JWT token"""
    access_token = request.headers.get("Authorization")
    if access_token and access_token.startswith("Bearer "):
        access_token = access_token.split(" ", 1)[1]
    else:
        access_token = None
  
    print(f"access_token: {access_token}")   
    
    streamlit_url = f"http://localhost:8501?session_token={access_token}"
    
    return JSONResponse({
        "success": True,
        "redirect_url": streamlit_url,
      
    })
@app.post("/api/email-message/")
async def chat_api_message2(request: Request):
    """Protected chat endpoint - redirects to Streamlit with session token"""
    token = request.cookies.get("access_token")
    msg = request.body
    from langchain_core.messages import  HumanMessage
    from prod_assistant.chatbot.langgraph_tool_backend import chatbot
    state = {
            'messages': [
                HumanMessage(
                    content="RE: URGENT: Critical Server Connection Issue with App Version 25.08.001 [ITR:0015942]?"
                )
            ]
        }
    # Add config with thread_id
    config = {
        "configurable": {
            "thread_id": token  # Unique identifier for this conversation
        }
    }
    print(state) 
    print("calling workflow chatbot!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Invoke with config
    result = chatbot.invoke(state, config=config)
    print(result['messages'][-1].content)
    return {"message" : result}


@app.get("/api/validate-session")
async def validate_session(session_token: str = Query(...)):
    """Validate session token from Streamlit"""
     
    print(f"Validating session token: {session_token}")
    
    access_token = active_sessions.get(session_token)
    session_data = jwt.decode(access_token, config["security"]["SECRET_KEY"], algorithms=[config["security"]["ALGORITHM"]])
    #  
    print(f"session_data: {session_data}")
    
    if not session_data:
        return {"valid": False}
    
    return {
        "valid": True,
        "username": session_data["username"],
        "user_id": session_data["user_id"],
        "jwt_token": session_data["jwt_token"]
    }
@app.post("/api/verify-token")
async def verify_user_token(token_data: dict = Depends(verify_token)):
    """Verify if token is valid"""
    return {
        "valid": True,
        "username": token_data.get("sub"),
        "user_id": token_data.get("user_id")
    }


@app.get("/api/protected-example")
async def protected_route(token_data: dict = Depends(verify_token)):
    """Example of a protected endpoint"""
    return {
        "message": "This is a protected endpoint",
        "user": token_data.get("sub"),
        "user_id": token_data.get("user_id")
    }

# ---------- Utility Endpoint (Remove in production) ----------
@app.post("/create-user")
async def create_user(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...)
):
    """Create a new user (for testing purposes)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        query = """
        INSERT INTO [user] (username, password_hash, email, created_at)
        VALUES (?, ?, ?, GETDATE())
        """
        cursor.execute(query, (username, password_hash, email))
        conn.commit()
        
        return {"message": "User created successfully"}
    except Exception as e:
        conn.rollback()
        raise HTTPException(
            status_code=400,
            detail=f"User creation failed: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

#==============Web API for Email Tracking
# Store active threads and websocket connections


active_threads: Dict[str, dict] = {}
websocket_connections: list[WebSocket] = []

# WebSocket for real-time notifications
@app.websocket("/ws/approvals")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

async def notify_pending_approval(approval_data: dict):
    """Send notification to all connected clients"""
    message = {
        "type": "approval_request",
        "data": approval_data
    }
    
    # Send to all connected WebSocket clients
    disconnected = []
    for connection in websocket_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        websocket_connections.remove(conn)

@app.post("/api/agent/process-email")
async def process_email(request: EmailRequest):
    """Start agent workflow for an email"""
    
    config = {
        "configurable": {
            "thread_id": request.email_id
        }
    }
    
    # Initial state
    initial_state = {
        "email_id": request.email_id,
        "sender_email" : request.sender_email,
        "email_content": request.email_content,
        "classification": request.classification,
        "draft_response": "",
        "approval_status": "pending"
    }
    
    # Store thread config
    active_threads[request.email_id] = {
        "config": config,
        "status": "processing"
    }
    
    # Run agent asynchronously
    asyncio.create_task(run_agent_workflow(initial_state, config, request.email_id))
    
    return {
        "status": "started",
        "email_id": request.email_id,
        "message": "Agent workflow initiated"
    }

async def run_agent_workflow(initial_state: dict, config: dict, email_id: str):
    """Run the agent workflow"""
    try:
        # Run until interrupt
        async for event in agent_app.astream(initial_state, config, stream_mode="values"):
            print(f"Agent event: {event}")
            
            # Check if we hit an interrupt
            state_snapshot = agent_app.get_state(config)
            if state_snapshot.next and "human_review" in state_snapshot.next:
                # We've hit the interrupt point
                print(f"Interrupt reached for {email_id}")
                
                # Get the interrupt value
                approval_data = pending_approvals.get(email_id, {}).get("data", {})
                
                # Notify frontend via WebSocket
                await notify_pending_approval(approval_data)
                
                # Update thread status
                active_threads[email_id]["status"] = "awaiting_approval"
                break
            elif "END" in state_snapshot.next :
                active_threads[email_id]["status"] = "send"
                print("\nFinal draft\n" + event['draft_response'])
                return event['draft_response']
        
    except Exception as e:
        print(f"Error in workflow: {str(e)}")
        active_threads[email_id]["status"] = "error"

@app.get("/api/agent/pending-approvals")
async def get_pending_approvals():
    """Get all pending approval requests"""
    pending = []
    
    for email_id, approval_info in pending_approvals.items():
        if approval_info["status"] == "pending":
            pending.append({
                "email_id": email_id,
                **approval_info["data"]
            })
    
    return {"pending_approvals": pending}

@app.get("/api/agent/approval-status/{email_id}")
async def get_approval_status(email_id: str):
    """Get status of specific approval request"""
    
    if email_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="Approval request not found")
    
    return pending_approvals[email_id]



@app.post("/api/agent/approve")
async def approve_response(approval: ApprovalResponse):
    """Submit approval decision and resume agent workflow using Command"""
    
    email_id = approval.email_id
    
    # Verify thread exists
    if email_id not in active_threads:
        raise HTTPException(status_code=404, detail="Email thread not found")
    
    if email_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="No pending approval for this email")
    
    # Get the config from active thread
    config = active_threads[email_id]["config"]
    
    try:
        # Create Command with resume to send back to the interrupt point
        human_response = Command(
            resume={
                "approved": approval.approved,
                "edited_response": approval.edited_response if approval.edited_response else "",
                "feedback": approval.feedback if approval.feedback else ""
            }
        )
        
        # Resume execution from the interrupt point
        result = agent_app.invoke(human_response, config)
        
        # result = None
        # async for event in agent_app.astream(
        #     human_response,
        #     config,
        #     stream_mode="values"
        # ):
        #     print(f"Agent resumed with event: {event}")
        #     result = event
        
        # Mark as resolved
        pending_approvals[email_id]["status"] = "resolved"
        active_threads[email_id]["status"] = "completed"
        if result:
            active_threads[email_id]["result"] = result
        
        return {
            "status": "success",
            "message": f"Email {email_id} {'approved' if approval.approved else 'rejected'}",
            "email_id": email_id,
            "approval_status": result.get("approval_status") if result else None
        }
        
    except Exception as e:
        pending_approvals[email_id]["status"] = "error"
        print(f"Error processing approval: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing approval: {str(e)}")
    
@app.post("/api/agent/approve/old")
async def approve_response_old(approval: ApprovalResponse):
    """Submit approval decision and resume agent workflow"""
    
    email_id = approval.email_id
    
    # Verify thread exists
    if email_id not in active_threads:
        raise HTTPException(status_code=404, detail="Email thread not found")
    
    if email_id not in pending_approvals:
        raise HTTPException(status_code=404, detail="No pending approval for this email")
    
    # Get the config
    config = active_threads[email_id]["config"]
    
    # Prepare approval decision
    approval_decision = {
        "approved": approval.approved,
        "edited_response": approval.edited_response,
        "feedback": approval.feedback
    }
    
    try:
        # Resume the agent with the approval decision
        # This updates the state and resumes from the interrupt
        agent_app.update_state(config, approval_decision, as_node="human_review")
        
        # Continue execution
        result = None
        async for event in agent_app.astream(approval_decision, config, stream_mode="values"):
            print(f"Agent resumed: {event}")
            result = event
        
        # Mark as resolved
        pending_approvals[email_id]["status"] = "resolved"
        active_threads[email_id]["status"] = "completed"
        if result:
            active_threads[email_id]["result"] = result
            
        return {
            "status": "success",
            "message": f"Email {email_id} {'approved' if approval.approved else 'rejected'}",
            "email_id": email_id,
            "final_state": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing approval: {str(e)}")

@app.get("/api/agent/threads")
async def get_active_threads():
    """Get all active agent threads"""
    return {"threads": active_threads}

@app.delete("/api/agent/thread/{email_id}")
async def cancel_thread(email_id: str):
    """Cancel an agent thread"""
    
    if email_id in active_threads:
        del active_threads[email_id]
    
    if email_id in pending_approvals:
        del pending_approvals[email_id]
    
    return {"status": "cancelled", "email_id": email_id}

# # Dummy implementations for other nodes
# def classify_email(state: EmailAgentState):
#     return {
#         "classification": {
#             "urgency": "high",
#             "intent": "support_request"
#         }
#     }

# def generate_draft(state: EmailAgentState):
#     return {
#         "draft_response": f"Thank you for your email regarding: {state['email_content'][:50]}..."
#     }

# def send_reply(state: EmailAgentState):
#     print(f"Sending reply for {state['email_id']}: {state['draft_response']}")
#     return state

if __name__ == "__main__":  
    uvicorn.run(app, host="0.0.0.0", port=8000)
