from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage

load_dotenv()

class AgentState(TypedDict):
    message: List[HumanMessage]

llm = ChatOpenAI(model="gpt-4o")

def process(state: AgentState) -> AgentState:
    """Processes the input message and generates a response using the LLM."""
    response = llm.invoke(state["message"])
    state["message"].append(response)
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.set_entry_point("process")
graph.set_finish_point("process")

app = graph.compile()
user_input = input("You: ")
while user_input.lower() != "exit":
    user_input = user_input.strip()
    result = app.invoke({"message": [HumanMessage(content=user_input)]})
    print(f"AI: {result['message'][-1].content}")
    user_input = input("You: ")
print("Conversation ended.")