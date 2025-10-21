#!/bin/bash
# Start Uvicorn in background
# uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
# streamlit run app/streamlit_frontend_tool.py.py --server.port 8501
EXPOSE 8000
python prod_assistant/mcp_servers/product_search_server.py 
uvicorn prod_assistant.router.main:app --host 0.0.0.0 --port 8000 --workers 2 

# Start Streamlit in foreground
# EXPOSE 8501
# streamlit run streamlit_frontend_tool.py --server.port 8501