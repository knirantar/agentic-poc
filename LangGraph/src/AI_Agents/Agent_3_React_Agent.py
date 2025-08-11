from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import tempfile
from PIL import Image
import os

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers together."""
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

tools = [add, subtract, multiply]
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

def process(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are a helpful assistant. Please answer the user's question.")
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": state["messages"] + [response]}  # Append AIMessage to state

def should_continue_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1] if messages else None

    # If user says "exit", stop.
    if isinstance(last_message, HumanMessage) and last_message.content.lower() == "exit":
        return END
    
    # If last message is from AI and requests a tool, go to tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tool_node"
    
    # If the last message is a tool's response, end after showing it
    if isinstance(last_message, ToolMessage):
        return END

    # Otherwise, default to END to avoid recursion
    return END


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_node("tool_node", ToolNode(tools=tools))
graph.add_edge(START, "process")
graph.add_conditional_edges("process", should_continue_node, {
    "process": "process",
    "tool_node": "tool_node",
    END: END
})
graph.add_edge("tool_node", "process")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, BaseMessage):
            message.pretty_print()
        else:
            print(message)

inputs = {"messages": [HumanMessage(content="Hello! Can you subtract 5 and 3 and multiply the result by 4?")]}
print_stream(app.stream(inputs, stream_mode="values"))
