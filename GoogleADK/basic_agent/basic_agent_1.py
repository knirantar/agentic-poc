# run_adk_agent.py
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

# Make sure OPENAI_API_KEY is set in env (or litellm configured)

# Wrap the provider model with LiteLlm so ADK can call OpenAI through LiteLLM
# Use whichever provider/model you want: "openai/gpt-3.5-turbo" or "openai/gpt-4o"
lite_model = LiteLlm(model="openai/gpt-3.5-turbo")

agent = LlmAgent(
    model=lite_model,                 # LiteLlm or model string
    name="my_openai_agent",
    description="Simple test agent that answers short prompts.",
    instruction="You are a helpful assistant. Keep answers short."
)

# Runner + session service (local, in-memory for dev)
session_service = InMemorySessionService()
APP_NAME = "local_app"
USER_ID = "user_1"
SESSION_ID = "session_1"

async def call_agent(prompt: str):
    # create session (async API)
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    # wrap user prompt as ADK Content
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    final_response = None
    # run_async yields events — iterate until we see the final response
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=content):
        # optionally print intermediate events for debugging:
        # print("EVENT:", event.author, event.is_final_response(), getattr(event.content, "parts", None))
        if event.is_final_response():
            # assume text is in the first part
            final_response = event.content.parts[0].text if event.content and event.content.parts else ""
            break

    print("Agent final response:", final_response)

if __name__ == "__main__":
    asyncio.run(call_agent("Tell me a short joke about AI and coffee."))
