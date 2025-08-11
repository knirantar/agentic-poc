from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    name: str
    age: int
    final: str

def first_node(state: AgentState) -> AgentState:
    state["final"] = f"Hello {state['name']}"
    return state

def second_node(state: AgentState) -> AgentState:
    state["final"] += f"! You are {state['age']} years old."
    return state
graph = StateGraph(AgentState)
graph.add_node("first", first_node)
graph.add_edge("first", "second")
graph.add_node("second", second_node)
graph.set_entry_point("first")
graph.set_finish_point("second")

app = graph.compile()
result = app.invoke({"name": "Nirantar", "age": 25})
print(result["final"])  # Output: Hello Nirant! You are 25 years old.