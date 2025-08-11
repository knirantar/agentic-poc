from typing import TypedDict, List
import random
from langgraph.graph import StateGraph, START, END
import tempfile
from PIL import Image
import os

class AgentState(TypedDict):
    name: str
    number: List[int]
    counter: int

def greeting_node(state: AgentState) -> AgentState:
    """A simple greeting node that initializes the state with a welcome message.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state with a greeting message.
    """
    state["name"] = f"Hello {state['name']}! How can I assist you today?"
    state["counter"] = 0
    return state

def random_number_node(state: AgentState) -> AgentState:
    """Generates a random number and updates the state.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state with a random number.
    """
    state["number"].append(random.randint(1, 100))
    state["counter"] += 1
    return state    

def should_continue_node(state: AgentState) -> AgentState:
    """Decides whether to continue generating random numbers based on the counter.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state with a decision on whether to continue.
    """
    if state["counter"] < 5:
        return "random_number"
    else:
        return END
    
graph = StateGraph(AgentState)
graph.add_node("greeting", greeting_node)
graph.add_node("random_number", random_number_node)
graph.add_edge("greeting", "random_number")
graph.add_conditional_edges("random_number", should_continue_node, {
    "random_number": "random_number",
    END: END
})

graph.set_entry_point("greeting")

app = graph.compile()
result = app.invoke({"name": "Nirantar", "number": []})
print(result)  # Output: Hello Nirantar! How can I assist you today?

# png_bytes = app.get_graph().draw_mermaid_png()

# # Save to a temporary file
# with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
#     tmp_file.write(png_bytes)
#     tmp_path = tmp_file.name

# # Open in default image viewer
# Image.open(tmp_path).show()

# # (Optional) delete temp file after opening
# os.remove(tmp_path)