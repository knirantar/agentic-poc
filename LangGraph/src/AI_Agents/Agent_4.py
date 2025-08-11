# agent_interactive.py
from typing import Annotated, Sequence, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
import os

load_dotenv()

# ----- State definition -----
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    turns: int

# ----- Toy document store and tools -----
document_content = ""

@tool
def update(content: str) -> str:
    """Updates the document content."""
    global document_content
    document_content += content + "\n"
    return "Document updated."

@tool
def save(filename: str) -> str:
    """Saves the document content to a file."""
    global document_content
    with open(filename, "w", encoding="utf-8") as f:
        f.write(document_content)
    return f"Document saved to {filename}."

# put your tool functions into a list (ToolNode expects this)
tools = [update, save]

# ----- LLM (we don't need to bind tools for our approach, but binding is ok) -----
# Note: binding isn't strictly required here because we'll call ToolNode explicitly,
# but it's harmless and sometimes helps the LLM format tool calls.
llm = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# ----- ToolNode object to execute tool calls when AI requests them -----
tool_node = ToolNode(tools)  # create once and reuse

# ----- system prompt describing agent behavior -----
SYSTEM_PROMPT = SystemMessage(content=(
    "You are Drafter, a helpful writing assistant. When you want to modify the document, "
    "emit a tool call to 'update' with the full content to add. When you want to persist "
    "the document, emit a tool call to 'save' with a file name. If you are done, respond with "
    "a final natural-language message and do not call any tools."
))

# ----- helper printers -----
def print_ai(msg: AIMessage):
    print("\n🤖 AI:", msg.content)
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print("🔧 AI requested tools:", [tc.get("name") for tc in msg.tool_calls])

def print_tool_results(messages: List[BaseMessage]):
    for m in messages:
        if isinstance(m, ToolMessage):
            print("\n🛠️ TOOL RESULT:", m.content)

# ----- interactive loop (external driver) -----
def run_interactive_agent():
    print("=== Drafter interactive agent ===")
    # initial empty conversation
    state_messages: List[BaseMessage] = []
    turns = 0
    max_turns = 10  # safety cap

    while True:
        # ask the user for an instruction
        user_text = input("\nYou: (type 'exit' to quit) > ").strip()
        if not user_text:
            print("Please type something or 'exit'.")
            continue
        if user_text.lower() == "exit":
            print("Exiting.")
            break

        # add user message to state
        user_msg = HumanMessage(content=user_text)
        state_messages.append(user_msg)

        # compose conversation and call LLM
        convo = [SYSTEM_PROMPT] + state_messages
        response: AIMessage = llm.invoke(convo)

        # append AI response
        state_messages.append(response)
        print_ai(response)

        # if AI requested tools, run ToolNode using current messages
        if hasattr(response, "tool_calls") and response.tool_calls:
            # ToolNode expects the state dict with "messages"
            tool_state = tool_node.invoke({"messages": state_messages})
            # tool_state is typically a dict containing "messages": [...]
            if "messages" in tool_state:
                state_messages = list(tool_state["messages"])
            else:
                # fallback: we keep same messages but warn
                print("Warning: tool_node returned unexpected state:", tool_state)

            # print any tool results that were appended
            print_tool_results(state_messages)

            # If last message is ToolMessage and indicates a save -> finish
            if isinstance(state_messages[-1], ToolMessage) and "saved" in state_messages[-1].content.lower():
                print("\n✅ Document saved. Ending session.")
                break

        # safety increment
        turns += 1
        if turns >= max_turns:
            print(f"\n🛑 Reached max turns ({max_turns}). Ending session to avoid loop.")
            break

    print("\n=== Session ended ===")


if __name__ == "__main__":
    run_interactive_agent()
