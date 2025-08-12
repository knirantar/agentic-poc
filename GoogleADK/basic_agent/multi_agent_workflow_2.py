import os
import asyncio
import httpx
from typing import List
from dotenv import load_dotenv

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or .env file")

API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-3.5-turbo"  # change to gpt-4 or another model if you prefer

HEADERS = {
    "Authorization": f"Bearer {OPENAI_KEY}",
    "Content-Type": "application/json",
}


async def call_openai(messages: List[dict], model: str = MODEL, max_tokens: int = 512) -> str:
    """Async call to OpenAI Chat Completions; returns assistant text."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        resp = await client.post(API_URL, json=payload, headers=HEADERS)
        resp.raise_for_status()
        j = resp.json()
        # defensive: pick first choice
        return j["choices"][0]["message"]["content"].strip()


# ---------------- Agent implementations ----------------
async def research_agent(topic: str) -> str:
    prompt = (
        "You are a concise research assistant. For the topic below, list 4 short factual "
        "bullet points (one line each). Use only verifiable high-level facts or widely-known statements.\n\n"
        f"Topic: {topic}\n\nBullet points:"
    )
    messages = [
        {"role": "system", "content": "You are a helpful research assistant."},
        {"role": "user", "content": prompt},
    ]
    return await call_openai(messages, max_tokens=300)


async def writer_agent(bullets_text: str) -> str:
    prompt = (
        "You are a clear, engaging writer. Given the research bullet points below, "
        "write a single informative paragraph (3-5 sentences) suitable for a general audience. "
        "Do not invent facts—use only what's in the bullets.\n\n"
        f"Research bullets:\n{bullets_text}\n\nParagraph:"
    )
    messages = [
        {"role": "system", "content": "You are a clear, engaging writer."},
        {"role": "user", "content": prompt},
    ]
    return await call_openai(messages, max_tokens=350)


async def critic_agent(paragraph: str) -> str:
    prompt = (
        "You are a constructive critic and editor. Improve the paragraph below for clarity, "
        "conciseness, and accuracy. If any claim seems speculative (not supported by the bullets), "
        "either flag it or rephrase cautiously. Provide the improved paragraph only.\n\n"
        f"Paragraph:\n{paragraph}\n\nImproved paragraph:"
    )
    messages = [
        {"role": "system", "content": "You are a critical editor who improves text while preserving facts."},
        {"role": "user", "content": prompt},
    ]
    return await call_openai(messages, max_tokens=300)


# ------------- Orchestration (sequential, streaming prints) -------------
async def run_pipeline(topic: str):
    print("\n=== Starting multi-agent pipeline ===\n")
    print(f"[INPUT TOPIC] {topic}\n")

    # Research
    print("-> ResearchAgent: gathering bullet points...")
    research_out = await research_agent(topic)
    print("\n[ResearchAgent output]\n")
    print(research_out.strip(), "\n")

    # Writer (consume research output)
    print("-> WriterAgent: writing paragraph from bullets...")
    writer_out = await writer_agent(research_out)
    print("\n[WriterAgent output]\n")
    print(writer_out.strip(), "\n")

    # Critic
    print("-> CriticAgent: improving the paragraph...")
    critic_out = await critic_agent(writer_out)
    print("\n[CriticAgent output]\n")
    print(critic_out.strip(), "\n")

    print("=== Pipeline finished ===\n")
    return {
        "research": research_out,
        "writer": writer_out,
        "critic": critic_out,
    }


if __name__ == "__main__":
    while True:
        try:
            topic = input("Enter a topic for the multi-agent pipeline (or 'exit' to quit): ").strip()
            if topic.lower() == 'exit':
                break
            asyncio.run(run_pipeline(topic))
        except Exception as e:
            print(f"Error: {e}\nPlease try again.")