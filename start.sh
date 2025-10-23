#!/bin/bash
python prod_assistant/mcp_servers/product_search_server.py &
uvicorn prod_assistant.router.main:app --host 0.0.0.0 --port 8000 --workers 2
