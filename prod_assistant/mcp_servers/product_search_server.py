from fastmcp import FastMCP

from retriever.retrieval import Retriever  
from langchain_community.tools import DuckDuckGoSearchRun
from retriever.conventional_retreiver import ConversationalRAG
import os


# Initialize MCP server
#mcp = FastMCP.from_fastapi(name="hybrid_search", app=app)
mcp = FastMCP(name="hybrid_search")
# Load retriever once
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# ---------- Helpers ----------
def format_docs(docs) -> str:
    """Format retriever docs into readable context."""
    if not docs:
        return ""
    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)
    return "\n\n---\n\n".join(formatted_chunks)

# ---------- MCP Tools ----------

#----------------------------
# adding FAISESS tool

@mcp.tool()
async def get_faiss_data(question: str) -> dict:
    """
    Fetch relevant documents from FAISS index and answer the question.
    If no relevant answer is found, increase top-k retrieval by 5 each time.
    """

    FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")
    FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
    session_id = "default"
    use_session_dirs = True
    
    index_dir = os.path.abspath(os.path.join(FAISS_BASE, session_id)) if use_session_dirs else FAISS_BASE
    print(index_dir)
    if not os.path.isdir(index_dir):
        return {"error": f"No FAISS index found at {index_dir}"}

    try:
        rag = ConversationalRAG(session_id=session_id)
        max_k = 50
        k = 10
        increment = 5

        while k <= max_k:
            print(f"Trying retrieval with top-k = {k}")
            await rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)
            response = rag.invoke(question, chat_history=[])

            # Customize this check based on your response format
            if response and isinstance(response, str) and "no relevant" not in response.lower():
                return {"answer": response, "retrieval_k": k}

            k += increment

        return {
            "error": "No relevant documents found after increasing k",
            "max_k_tried": max_k
        }

    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
async def get_traccs_email_info(query: str) -> str:
    """ This is tool to access data of email about Traccs product which is a community care and time recording system. It provides service information, 
    roster information, Attendance Time Recording, Payroll, Recipient Data, Staff and carer Information
    Retrieve email information of  product Traccs email a community care and time recording system emails conservation of support staff with clients
    Search for a given query from local retriever extract answer of query and reply with formal email to the question asked in email.
    at the end of message add a thank you note and signature with name as Traccs chatbot.
    """
    try:
        print("get_traccs_email_info called with query:", query)
        # retriever =  retriever_obj.load_retriever()
        if not retriever:
            return "Retriever not initialized."
        docs =  retriever.invoke(query)
        context = format_docs(docs)
        if not context.strip():
            return "No local results found."
        return context
    except Exception as e:
        return f"Error retrieving product info: {str(e)}"

@mcp.tool()
async def get_product_info(query: str) -> str:
    """Retrieve product information for a given query from local retriever."""
    try:
        print("get_product_info called with query:", query)
        if not retriever:
            return "Retriever not initialized."
        docs =  retriever.invoke(query)
        context = format_docs(docs)
        if not context.strip():
            return "No local results found."
        return context
    except Exception as e:
        return f"Error retrieving product info: {str(e)}"

@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web using DuckDuckGo if retriever has no results."""
    try:
       
        duckduckgo =DuckDuckGoSearchRun(region="us-en")
        return duckduckgo.run(query)
    except Exception as e:
        return f"Error during web search: {str(e)}"
    
@mcp.tool()
async def chat(msg: str) -> str:

    """Chat tool using Agentic RAG workflow with MCP web search and FAISS retriever."""
       
    # Import your workflow here to avoid circular imports
    from workflow.agentic_workflow_with_mcp_websearch import AgenticRAG
    
    rag_agent = AgenticRAG()
    answer = await rag_agent.run(msg)
    print(f"Agentic Response: {answer}")
    return answer
# ---------- Run Server ----------
if __name__ == "__main__":
    # mcp.run(transport="stdio")
    # mcp.run(transport="streamable-http")   

    config = {
        
        "host": os.getenv("MCP_HOST", "127.0.0.1"),
        "port": int(os.getenv("MCP_PORT", 8000))        
        
    }
    mcp.run(transport="streamable-http", **config)
