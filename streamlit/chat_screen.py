import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from chatbot.langgraph_tool_backend2 import chatbot, retrieve_all_threads


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history at the top
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Display files if present in message
        if "files" in message:
            st.markdown("**📎 Attached Files:**")
            for file_info in message["files"]:
                file_size_kb = file_info["size"] / 1024
                st.markdown(f"- **{file_info['name']}** ({file_info['type']}, {file_size_kb:.2f} KB)")
        
        if "status" in message:
            st.status(message["status"], state="complete", expanded=False)

# Handle menu actions (shown above chat input)
if "action" in st.session_state:
    if st.session_state["action"] == "ingest":
        with st.expander("📥 Ingest File", expanded=True):
            uploaded_file = st.file_uploader("Choose a file to ingest", key="file_uploader")
            if uploaded_file:
                if st.button("Process File"):
                    st.success(f"Ingesting: {uploaded_file.name}")
                    # Add your ingestion logic here
                    # Example: process_file(uploaded_file)
                    del st.session_state["action"]
                    st.rerun()
            if st.button("Cancel", key="cancel_ingest"):
                del st.session_state["action"]
                st.rerun()
    
    elif st.session_state["action"] == "scrape":
        with st.expander("🔍 Scrape File", expanded=True):
            url = st.text_input("Enter URL to scrape", key="scrape_url")
            if st.button("Start Scraping"):
                if url:
                    st.success(f"Scraping: {url}")
                    # Add your scraping logic here
                    # Example: scrape_url(url)
                    del st.session_state["action"]
                    st.rerun()
                else:
                    st.error("Please enter a valid URL")
            if st.button("Cancel", key="cancel_scrape"):
                del st.session_state["action"]
                st.rerun()
    
    elif st.session_state["action"] == "save":
        with st.expander("💾 Save to DB", expanded=True):
            st.write("Save current conversation data to database?")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Confirm Save", use_container_width=True):
                    st.success("Saving data to database...")
                    # Add your DB save logic here
                    # Example: save_to_database(st.session_state.messages)
                    del st.session_state["action"]
                    st.rerun()
            with col_b:
                if st.button("Cancel", key="cancel_save", use_container_width=True):
                    del st.session_state["action"]
                    st.rerun()

# Create a fixed container at the bottom for chat input with menu
st.markdown("---")

# Use custom CSS to fix input at bottom
st.markdown("""
<style>
    .main > div:last-child {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: var(--background-color);
        padding: 1rem;
        z-index: 999;
        border-top: 1px solid var(--border-color);
    }
    
    .main {
        padding-bottom: 150px;
    }
    
    /* Style the menu button to look integrated */
    [data-testid="column"]:first-child button {
        height: 3rem;
        font-size: 1.5rem;
        border-radius: 0.5rem 0 0 0.5rem;
    }
    
    /* Adjust chat input styling */
    [data-testid="stChatInput"] {
        border-radius: 0 0.5rem 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

input_container = st.container()

with input_container:
    # Create columns for menu button and chat input
    menu_col, input_col = st.columns([1, 11])
    
    with menu_col:
        # Plus button menu
        with st.popover("⊕", use_container_width=True):
            st.markdown("**Actions**")
            
            # Ingest File option
            if st.button("📥 Ingest File", use_container_width=True, key="menu_ingest"):
                st.session_state["action"] = "ingest"
                st.rerun()
            
            # Scrape File option
            if st.button("🔍 Scrape File", use_container_width=True, key="menu_scrape"):
                st.session_state["action"] = "scrape"
                st.rerun()
            
            # Save to DB option
            if st.button("💾 Save to DB", use_container_width=True, key="menu_save"):
                st.session_state["action"] = "save"
                st.rerun()
    
    with input_col:
        user_input = st.chat_input(
            "Type your message here...", 
            key="chat_input",
            accept_file=True,
            max_file_size=10  # Maximum file size in MB
        )

# Process user input if provided
if user_input:
    # Check if user_input contains text or files
    message_text = ""
    uploaded_files = []
    
    # Handle dictionary response (when files are uploaded)
    if isinstance(user_input, dict):
        message_text = user_input.get("text", "")
        uploaded_files = user_input.get("files", [])
    else:
        # Handle string response (text only)
        message_text = user_input
    
    # Add user message to history
    user_message = {
        "role": "user", 
        "content": message_text,
    }
    
    # Add file information if files were uploaded
    if uploaded_files:
        user_message["files"] = [
            {
                "name": f.name,
                "type": f.type,
                "size": f.size
            } for f in uploaded_files
        ]
    
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.write(message_text)
        
        # Display uploaded files
        if uploaded_files:
            st.markdown("**📎 Attached Files:**")
            for file in uploaded_files:
                file_size_kb = file.size / 1024
                st.markdown(f"- **{file.name}** ({file.type}, {file_size_kb:.2f} KB)")
                
                # Process different file types
                if file.type.startswith("image/"):
                    st.image(file, caption=file.name, use_container_width=True)
                elif file.type == "text/plain":
                    content = file.read().decode("utf-8")
                    with st.expander(f"View {file.name}"):
                        st.text(content)
                elif file.type == "application/pdf":
                    st.info(f"PDF file uploaded: {file.name}")
                    # Add PDF processing logic here
                else:
                    st.info(f"File uploaded: {file.name}")
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
        status_holder = {"box": None}
        
        # Prepare context with file information
        context_message = message_text
        if uploaded_files:
            file_info = ", ".join([f.name for f in uploaded_files])
            context_message = f"{message_text}\n[User uploaded files: {file_info}]"

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=context_message)]},
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
    
    # Add assistant message to history
    status_text = "✅ Tool finished" if status_holder["box"] is not None else None
    st.session_state.messages.append({
        "role": "assistant", 
        "content": ai_message,
        **({"status": status_text} if status_text else {})
    })
    
    st.rerun()