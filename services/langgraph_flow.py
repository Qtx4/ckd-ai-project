from langgraph.graph import StateGraph
from services.rag_service import ask_ckd_bot

def node(state):
    question = state["question"]
    answer = ask_ckd_bot(question)
    return {"answer": answer}

graph = StateGraph(dict)
graph.add_node("ckd_node", node)
graph.set_entry_point("ckd_node")
graph.set_finish_point("ckd_node")

app = graph.compile()