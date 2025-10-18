from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from prompt_library.prompts import PROMPT_REGISTRY, PromptType
from retriever.retrieval import Retriever
from utils.model_loader import ModelLoader
from evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

class AgenticRAG:
    """Agentic RAG pipeline using LangGraph + MCP (Retriever + WebSearch)."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # ---------- Initialization ----------
    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()

        # Initialize MCP client
        self.mcp_client = MultiServerMCPClient(
            {
                "hybrid_search": {
                    "transport": "streamable_http",
                    "url": "http://localhost:8000/mcp"
                }
            }
        )
       
        # Build workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

        # Load MCP tools asynchronously
        asyncio.run(self._safe_async_init())

    async def async_init(self):
        """Load MCP tools asynchronously."""
        self.mcp_tools = await self.mcp_client.get_tools()

    async def _safe_async_init(self):
        """Safe async init wrapper (prevents event loop crash)."""
        try:
            self.mcp_tools = await self.mcp_client.get_tools()
            print("MCP tools loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load MCP tools — {e}")
            self.mcp_tools = []

    # ---------- Nodes ----------
    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        messages = state["messages"]
        last_message = messages[-1].content

        if any(word in last_message.lower() for word in ["price", "review", "product", "email","conversation","issue","fix","guide", "details", "information", "specification", "specifications", "features", "feature"]) :
        #, "email","conversation","issue","fix","guide", "details", "information", "specification", "specifications", "features", "feature"
            return {"messages": [HumanMessage(content="TOOL: retriever : " + last_message)]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message}) or "I'm not sure about that."
            return {"messages": [HumanMessage(content=response)]}
    def chat_node(self,state: AgentState):
        """LLM node that may answer or request a tool call."""
        messages = state["messages"]
        self.llm_with_tools = self.llm.bind_tools(self.mcp_tools)  
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    def format_docs(self,docs) -> str:
        if not docs:
            return "No relevant documents found."
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

    async def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER (MCP) ---")
        query = state["messages"][-1].content

        tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
        if not tool:
            return {"messages": [HumanMessage(content="Retriever tool not found in MCP client.")]}

        try:
            result = await tool.ainvoke({"query": query})
            context = result or "No relevant product data found."
        except Exception as e:
            context = f"Error invoking retriever: {e}"

        return {"messages": [HumanMessage(content=context)]}


    async def _vector_retriever_new(self, state: AgentState):
        print("--- RETRIEVER (MCP) ---")
        query = state["messages"][-1].content
        print("Query to MCP:", query)
        
        if not self.mcp_tools:
            return {"messages": [HumanMessage(content="No MCP tools available.")]}
        
        try:
            print("--- BINDING TOOLS TO LLM ---")
            # Bind all available MCP tools to the LLM
            llm_with_tools = self.llm.bind_tools(self.mcp_tools)
            
            # Let the LLM decide which tool(s) to use
            response = await llm_with_tools.ainvoke(query)
            print("LLM Response:", response[-1].content)
            # Check if the LLM wants to call any tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"--- LLM DECIDED TO CALL: {[tc['name'] for tc in response.tool_calls]} ---")
                
                # Execute the tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # Find the matching tool
                    tool = next((t for t in self.mcp_tools if t.name == tool_name), None)
                    print(f"Calling tool: {tool_name} with args: {tool_args}")
                    if tool:
                        try:
                            print(f"--- EXECUTING {tool_name} ---")
                            result = await tool.ainvoke(tool_args)
                            tool_results.append(result)
                        except Exception as e:
                            tool_results.append(f"Error calling {tool_name}: {e}")
                    else:
                        tool_results.append(f"Tool {tool_name} not found")
                
                # Combine all tool results
                context = "\n\n".join(str(r) for r in tool_results) if tool_results else "No results from tools."
                return {"messages": [HumanMessage(content=context)]}
            else:
                # LLM didn't call any tools, return its response directly
                print("--- LLM RESPONDED WITHOUT CALLING TOOLS ---")
                return {"messages": [AIMessage(content=response.content)]}
                
        except Exception as e:
            print(f"Error in retriever: {e}")
            return {"messages": [HumanMessage(content=f"Error invoking retriever: {e}")]}
    
    async def _web_search(self, state: AgentState):
        print("--- WEB SEARCH (MCP) ---")
        query = state["messages"][-1].content
        tool = next(t for t in self.mcp_tools if t.name == "web_search")
        result = await tool.ainvoke({"query": query})  # ✅
        context = result if result else "No data from web"
        return {"messages": [HumanMessage(content=context)]}


    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs}) or ""
        return "generator" if "yes" in score.lower() else "rewriter"

    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            response = chain.invoke({"context": docs, "question": question}) or "No response generated."
        except Exception as e:
            response = f"Error generating response: {e}"

        return {"messages": [HumanMessage(content=response)]}

    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["messages"][0].content

        prompt = ChatPromptTemplate.from_template(
            "Rewrite this user query to make it more clear and specific for a search engine. "
            "Do NOT answer the query. Only rewrite it.\n\nQuery: {question}\nRewritten Query:"
        )
        chain = prompt | self.llm | StrOutputParser()

        try:
            new_q = chain.invoke({"question": question}).strip()
        except Exception as e:
            new_q = f"Error rewriting query: {e}"

        return {"messages": [HumanMessage(content=new_q)]}

    # ---------- Build Workflow ----------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("WebSearch", self._web_search)

        # Workflow edges
        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "WebSearch")
        workflow.add_edge("WebSearch", "Generator")

        return workflow

    # ---------- Public Run ----------
    async def run(self, query: str, thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content

# ---------- Standalone Test ----------
if __name__ == "__main__":
    rag_agent = AgenticRAG()
    query =  "In email What issue was faced by MTA app while adding Travel Claim?"
    answer = asyncio.run( rag_agent.run(query))

    print("\nFinal Answer:\n", answer)
