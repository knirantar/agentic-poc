from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

load_dotenv()

# LLM setup
llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini") 

def langchain_agent():
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)

    # Updated AgentType
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,  # Still works but flagged for LangGraph migration
        verbose=True
    )

    query = (
        "Retrieve the complete list of all Presidents of the United States from the most reliable sources.  "
        "For each president, include:"
        "1. Full name  "
        "2. Term start year and end year  "
        "3. Political party  "
        "Present the output as a clean, ordered table in chronological order starting from George Washington to the current president.  "
        "If partial information is found, fill in missing details by reasoning step-by-step and cross-checking with historical context."
    )

    result = agent.run(query)
    print(f"Agent Result: {result}")


if __name__ == "__main__":
    langchain_agent()