from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """Processes the input message and generates a response using the LLM."""
    response = llm.invoke(state["message"])
    state["message"].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")  # Debugging output
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.set_finish_point("process")
app = graph.compile()

conversation_history = []

user_input = input("You: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    user_input = user_input.strip()
    result = app.invoke({"message": conversation_history})
    conversation_history = result["message"]
    user_input = input("You: ")
print("Conversation ended.")

with open("conversation_history.txt", "w") as f:
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            f.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            f.write(f"AI: {message.content}\n")

print("Conversation history saved to conversation_history.txt")  # Confirmation message