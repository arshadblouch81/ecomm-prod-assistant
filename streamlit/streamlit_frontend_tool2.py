
import asyncio
import requests
import streamlit as st
from chatbot.langgraph_tool_backend2 import chatbot, retrieve_all_threads, async_initialize
from prod_assistant.etl.data_ingestion import DataIngestion
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
import uuid
from dotenv import load_dotenv

load_dotenv()

#******************************** Session and token Management *********************************
# Get session token from URL
query_params = st.query_params
session_token = query_params.get("session_token", None)

# Validate session with FastAPI
# if session_token:
#     response = requests.get(
#         "http://localhost:8000/api/validate-session",
#         params={"session_token": session_token}
#     )
    
#     if response.json().get("valid"):
#         session_data = response.json()
        
#         # Store in session state
#         if "authenticated" not in st.session_state:
#             st.session_state.authenticated = True
#             st.session_state.username = session_data["username"]
#             st.session_state.user_id = session_data["user_id"]
#             st.session_state.jwt_token = session_data["jwt_token"]
#     else:
        # st.error("Invalid or expired session")
        # st.stop()
# else:
#     st.error("No session token provided")
#     st.stop()
# **************************************** utility functions *************************

# using langraph for chatbot
# Your Streamlit Chat Interface
st.title("Adamas Chatbot")
async_initialize()
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        

def load_conversation(thread_id):       
    try:
        if (chatbot):
            return chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages'] 
    except KeyError:
        return []
    
def load_first_message(thread_id):
    if (chatbot):
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])
    
        if messages:
            return messages[0].content
        else:
            return 'New Chat of ' + str(thread_id)

def init_frontend():
    # **************************************** Session Setup ******************************
    if 'message_history' not in st.session_state:
        st.session_state['message_history'] = []

    if 'thread_id' not in st.session_state:
        st.session_state['thread_id'] = generate_thread_id()

    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = retrieve_all_threads()

    add_thread(st.session_state['thread_id'])


    # **************************************** Sidebar UI *********************************
    # if 'sidebar_buttons' not in st.session_state:
    #     st.session_state['sidebar_buttons'] = []

        



    if st.sidebar.button('New Chat'):  
        reset_chat()


    for thread_id in st.session_state['chat_threads'][::-1]:
        btn_label =  load_first_message(thread_id)
        if st.sidebar.button(btn_label, key=thread_id):
            st.session_state['thread_id'] = thread_id
            messages = load_conversation(thread_id)

            temp_messages = []

            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role='user'
                else:
                    role='assistant'
                temp_messages.append({'role': role, 'content': msg.content})

            st.session_state['message_history'] = temp_messages


    # **************************************** st UI ************************************


    # loading the conversation history
    for message in st.session_state['message_history']:
        with st.chat_message(message['role']):
            st.text(message['content'])
            if "status" in message:
                st.status(message["status"], state="complete", expanded=False)

                        
    user_input = st.chat_input('Type here')
    

    # Process user input if provided
    if user_input:

        # first add the message to message_history
        st.session_state['message_history'].append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.text(user_input)
            # if (len(st.session_state['message_history']) == 1):
            #     st.session_state['message_history'][0]['content'] = load_first_message(st.session_state['thread_id'])
    # CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

        CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']},
            'metadata' :{
                'thread_id': st.session_state['thread_id']              
            },
            'run_name' : 'chat_turn',
            'recursion_limit': 50
            }


    # Assistant streaming block
        with st.chat_message("assistant"):
            # Use a mutable holder so the generator can set/modify it
            status_holder = {"box": None, "menu_container": None}

            def ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                    
                ):
                    # Lazily create & update the SAME status container when any tool runs
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` …", expanded=True
                            )
                        
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` …",
                                state="running",
                                expanded=True,
                            )

                    # Stream ONLY assistant tokens
                    if isinstance(message_chunk, AIMessage):
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

            # Finalize only if a tool was actually used
            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

            st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
            
init_frontend()