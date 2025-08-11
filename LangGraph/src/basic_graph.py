from typing import Dict, TypedDict
from langgraph.graph import StateGraph
from IPython.display import display, Image
import tempfile
from PIL import Image
import os


class AgentState(TypedDict):
    messages: str

def greeting_node(state: AgentState) -> AgentState:
    """A simple greeting node that initializes the state with a welcome message.

    Args:
        state (AgentState): The current state of the agent.

    Returns:
        AgentState: The updated state with a greeting message.
    """
    state["messages"] = "Hello! "+ state["messages"] + " How can I assist you today?"
    return state

graph = StateGraph(AgentState)
graph.add_node("greeting", greeting_node)
graph.set_entry_point("greeting")
graph.set_finish_point("greeting")

app = graph.compile()
result = app.invoke({"messages":"Nice to meet you!"})
print(result["messages"])
# # Get the image bytes from Mermaid
# png_bytes = app.get_graph().draw_mermaid_png()

# # Save to a temporary file
# with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
#     tmp_file.write(png_bytes)
#     tmp_path = tmp_file.name

# # Open in default image viewer
# Image.open(tmp_path).show()

# # (Optional) delete temp file after opening
# os.remove(tmp_path)