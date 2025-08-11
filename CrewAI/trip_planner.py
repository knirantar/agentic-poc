from textwrap import dedent
from crewai import Agent, Task, Crew
from crewai.tools import tool
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Make sure you set your API key in the environment before running:
# export OPENAI_API_KEY="your_api_key_here"
load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tools

@tool("Calculator")
def calculator_tool(expression: str) -> str:
    """Perform arithmetic calculations (basic math only)."""
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# Agents
travel_agent = Agent(
    role="Travel Agent",
    goal="Create the most amazing travel itineraries with budget and packing suggestions for your clients to travel to any destination.",
    backstory="Specialist in travel planning and logistics with decades of experience",
    tools=[calculator_tool],
    llm=llm,
    verbose=True
)

city_selection_expert = Agent(
    role="City Selection Expert",
    goal="Select the best city for the client's trip based on budget, climate, and preferences.",
    backstory="Geography expert with a knack for finding hidden gems.",
    tools=[],
    llm=llm,
    verbose=True
)

local_tour_guide = Agent(
    role="Local Tour Guide",
    goal="Provide detailed day-by-day itineraries for the chosen city.",
    backstory="Experienced guide who knows every corner of the city.",
    tools=[],
    llm=llm,
    verbose=True
)

quality_control_expert = Agent(
    role="Quality Control Expert",
    goal="Ensure the travel plan meets client expectations and standards.",
    backstory="Perfectionist with attention to detail in planning.",
    tools=[],
    llm=llm,
    verbose=True
)

manager = Agent(
    role="Manager",
    goal="Oversee the full travel planning process and coordinate between agents.",
    backstory="Seasoned travel project manager.",
    tools=[],
    llm=llm,
    verbose=True
)

# Tasks
manager_task = Task(
    description=dedent("""
    Oversee the trip planning process, ensuring each specialist delivers their part
    and all components form a coherent and exciting itinerary.
    """),
    expected_output="A fully coordinated travel plan with all sections integrated.",
    agent=manager
)

travel_task = Task(
    description=dedent("""
    Prepare a travel itinerary including budget breakdown and packing list for the trip.
    """),
    expected_output="Complete itinerary with budget and packing suggestions.",
    agent=travel_agent
)

city_task = Task(
    description=dedent("""
    Choose the best city based on preferences and constraints.
    """),
    expected_output="Chosen city with a short rationale.",
    agent=city_selection_expert
)

tour_task = Task(
    description=dedent("""
    Create a detailed, day-by-day tour plan for the chosen city.
    """),
    expected_output="Daily tour plan with timings and activity descriptions.",
    agent=local_tour_guide
)

quality_task = Task(
    description=dedent("""
    Review the travel plan to ensure it is complete, practical, and appealing.
    """),
    expected_output="Approval or a list of improvements.",
    agent=quality_control_expert
)

# Crew
crew = Crew(
    agents=[manager, travel_agent, city_selection_expert, local_tour_guide, quality_control_expert],
    tasks=[manager_task, travel_task, city_task, tour_task, quality_task],
    verbose=True
)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
