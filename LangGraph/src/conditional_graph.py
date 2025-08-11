from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import tempfile
from PIL import Image
import os

class AgentState(TypedDict):
    number1: int
    number2: int
    operation: str
    final_number: int

def add(state: AgentState) -> AgentState:
    """Adds two numbers."""
    state["final_number"] = state["number1"] + state["number2"]
    return state

def subtract(state: AgentState) -> AgentState:
    """Subtracts two numbers."""
    state["final_number"] = state["number1"] - state["number2"]
    return state    

def multiply(state: AgentState) -> AgentState:
    """Multiplies two numbers."""
    state["final_number"] = state["number1"] * state["number2"]
    return state

def divide(state: AgentState) -> AgentState:
    """Divides two numbers."""
    if state["number2"] == 0:
        raise ValueError("Cannot divide by zero.")
    state["final_number"] = state["number1"] / state["number2"]
    return state

def decide_operation(state: AgentState) -> AgentState:
    """Decides which operation to perform based on the operation field."""
    if state["operation"] == "add":
        return "add"
    elif state["operation"] == "subtract":
        return "subtract"
    elif state["operation"] == "multiply":
        return "multiply"
    elif state["operation"] == "divide":
        return "divide"
    else:
        raise ValueError("Invalid operation specified.")
    
# Create a state graph for the calculator
# This graph will have multiple nodes for different operations
# and will decide which operation to perform based on the input.
graph = StateGraph(AgentState)
graph.add_node("router", lambda state: state)
graph.add_node("add", add)
graph.add_node("subtract", subtract)
graph.add_node("multiply", multiply)
graph.add_node("divide", divide)
graph.add_edge(START, "router")
graph.add_conditional_edges("router", decide_operation, {
    "add": "add",
    "subtract": "subtract",
    "multiply": "multiply",
    "divide": "divide"
})
graph.add_edge("add", END)
graph.add_edge("subtract", END)
graph.add_edge("multiply", END)
graph.add_edge("divide", END)

app = graph.compile()
result = app.invoke({
    "number1": 10,
    "number2": 5,
    "operation": "divide"
})

print(result["final_number"])  # Output: 15

# Get the image bytes from Mermaid
png_bytes = app.get_graph().draw_mermaid_png()

# Save to a temporary file
with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
    tmp_file.write(png_bytes)
    tmp_path = tmp_file.name

# Open in default image viewer
Image.open(tmp_path).show()

# (Optional) delete temp file after opening
os.remove(tmp_path)