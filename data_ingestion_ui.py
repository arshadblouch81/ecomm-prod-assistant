
import streamlit as st
from prod_assistant.etl.data_ingestion import DataIngestion
from dotenv import load_dotenv

load_dotenv()
# **************************************** Data Ingestion functions *************************


# **************************************** Sidebar UI *********************************
# if 'sidebar_buttons' not in st.session_state:
#     st.session_state['sidebar_buttons'] = []
st.set_page_config(layout="wide")
st.sidebar.title("📂 Product Assistant - Data Ingestion")

# **************************************** Main UI ************************************

# Spacer to push input to bottom
st.markdown("<br>" * 2, unsafe_allow_html=True)


# Handle menu actions
if "action" in st.session_state:
    if st.session_state["action"] == "ingest":
        with st.expander("📥 Ingest File", expanded=True):
            uploaded_file = st.file_uploader("Choose a file to ingest", key="file_uploader")
            if uploaded_file:
                if st.button("Process File"):
                    st.success(f"Ingesting: {uploaded_file.name}")
                    with st.spinner("📡 Initializing ingestion pipeline..."):
                        try:
                            ingestion = DataIngestion()
                            st.info("🚀 Running ingestion pipeline...")
                            ingestion.save_data([uploaded_file])
                            st.success("✅ Data successfully ingested to AstraDB!")
                        except Exception as e:
                            st.error("❌ Ingestion failed!")
                            st.exception(e)
                        # del st.session_state["action"]
                        # st.rerun()
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
                    # Example: save_to_database(chat_history)
                    del st.session_state["action"]
                    st.rerun()
            with col_b:
                if st.button("Cancel", key="cancel_save", use_container_width=True):
                    del st.session_state["action"]
                    st.rerun()

input_container = st.container()


with input_container:
    # Create menu button and chat input
   
    # Plus button menu
    with st.popover("⊕ Actions", use_container_width=True):
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
    
   