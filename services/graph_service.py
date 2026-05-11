from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional

from services.llm_service import get_llm_response
from services.rag_service import retrieve_docs


# -----------------------------
# STATE
# -----------------------------
class State(TypedDict):
    query: str
    response: Optional[str]


# -----------------------------
# NODE
# -----------------------------
def assistant_node(state: State):

    query = state.get("query")

    if not query:
        return {"response": "⚠️ No question received"}

    docs = retrieve_docs(query)
    response = get_llm_response(query, docs)

    return {"response": response}   # ✅ FIXED


# -----------------------------
# GRAPH
# -----------------------------
graph = StateGraph(State)

graph.add_node("assistant", assistant_node)
graph.set_entry_point("assistant")
graph.add_edge("assistant", END)

chain = graph.compile()