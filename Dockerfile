FROM python:3.12.10

WORKDIR /app

# install git
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY prod_assistant ./prod_assistant

RUN pip install --no-cache-dir -r requirements.txt

COPY . .


EXPOSE 8000
COPY start.sh ./start.sh
CMD ["./start.sh"]



# CMD ["bash", "-c", "python prod_assistant/mcp_servers/product_search_server.py & uvicorn prod_assistant.router.main:app --host 0.0.0.0 --port 8000 --workers 2 "]


# EXPOSE 8501
# CMD ["bash", "-c", "streamlit run streamlit_frontend_tool.py"]