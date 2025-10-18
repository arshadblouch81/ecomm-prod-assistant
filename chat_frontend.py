# streamlit_frontend_tool.py

import streamlit as st
from chatbot.langgraph_tool_backend2 import (
    chatbot, 
    retrieve_all_threads, 
    load_first_message
)
from langchain_core.messages import HumanMessage
import uuid

st.set_page_config(
    page_title="E-commerce Product Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if 'messages' not in st.session_state:
    st.session_state.messages = []


def send_message(message: str, thread_id: str):
    """Send message to chatbot and get response"""
    try:
        state = {'messages': [HumanMessage(content=message)]}
        config = {'configurable': {'thread_id': thread_id}}
        
        # Get response
        result = chatbot.invoke(state, config=config)
        
        # Extract last message
        if result and 'messages' in result:
            return result['messages'][-1].content
        return "No response generated"
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


# Sidebar - Thread Management
with st.sidebar:
    st.title("💬 Conversations")
    
    # New Chat button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # List all threads
    threads = retrieve_all_threads()
    
    if threads:
        st.subheader("Recent Chats")
        for thread in threads:
            first_msg = load_first_message(thread)
            # Truncate long messages
            display_text = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
            
            if st.button(
                display_text, 
                key=thread,
                use_container_width=True,
                type="secondary" if thread != st.session_state.thread_id else "primary"
            ):
                st.session_state.thread_id = thread
                st.session_state.messages = []
                st.rerun()
    else:
        st.info("No conversations yet")
    
    st.divider()
    
    # Display current thread ID
    st.caption(f"Current Thread: {st.session_state.thread_id[:8]}...")


# Main Chat Interface
st.title("🤖 E-commerce Product Assistant")
st.caption("Ask questions about products, orders, or technical issues")

# Load existing messages for current thread
if not st.session_state.messages:
    try:
        state = chatbot.get_state(
            config={'configurable': {'thread_id': st.session_state.thread_id}}
        )
        existing_messages = state.values.get('messages', [])
        
        for msg in existing_messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            st.session_state.messages.append({
                "role": role,
                "content": msg.content
            })
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = send_message(prompt, st.session_state.thread_id)
            
            if response:
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
            else:
                st.error("Failed to get response")

# Footer
st.divider()
st.caption("💡 Tip: Use the sidebar to switch between conversations or start a new chat")