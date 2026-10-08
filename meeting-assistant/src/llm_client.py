import json
from openai import OpenAI
from pydantic import BaseModel
from src.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

# Initialize the OpenRouter client
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
)

def call_llm_structured(prompt: str, system_prompt: str, model: str, response_format: type) -> BaseModel:
    """
    Calls the LLM and forces the output to match a Pydantic schema using Structured Outputs.
    """
    try:
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=0.1, # Keep it deterministic
        )
        return completion.choices[0].message.parsed
    except Exception as e:
        print(f"LLM Call Failed: {e}")
        return None

def call_llm_text(prompt: str, system_prompt: str, model: str) -> str:
    """Standard LLM call returning raw text."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM Call Failed: {e}")
        return ""