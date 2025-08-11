from typing import List, TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    values: List[int]
    name: str
    result: str

def process_values_node(state: AgentState) -> AgentState:
    """Processes a list of integers and returns their sum along with a greeting.

    Args:
        state (AgentState): The current state of the agent containing values and name.

    Returns:
        AgentState: The updated state with the sum and greeting message.
    """
    print(state)
    state["result"] = f"Hi {state["name"]} your sum is : {sum(state["values"])}"
    print(state)
    return state

graph = StateGraph(AgentState)
graph.add_node("process_values", process_values_node)
graph.set_entry_point("process_values")
graph.set_finish_point("process_values")

app = graph.compile()
answer = app.invoke({
    "values": [1, 2, 3, 4, 5],
    "name": "Nirantark Kulkarni",
})

print(answer["result"])  # Output: Hi Nirant your sum is : 15