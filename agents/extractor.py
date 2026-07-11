"""
Extractor Agent
===============
Extracts explicit factual details (such as names, ages, taglines, locations, etc.)
provided by the user in their prompt. Stored as a structured dictionary.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),  # type: ignore[arg-type]
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.0,  # Zero temperature for deterministic extraction
    )

EXTRACTOR_SYSTEM_PROMPT = """\
You are an expert information extraction assistant.
Your task is to analyze the user's website request and extract any explicit, specific factual details that the user has provided.

Specifically, look for facts such as:
- Name (e.g., name is "Rakesh")
- Age (e.g., age is "19")
- Tagline (e.g., tagline is "Aspiring Government Officer")
- Any other specific values the user explicitly typed for their profile or website.

RULES:
1. Extract the facts exactly as provided in the prompt. Do not modify, correct, or "improve" values (e.g. if the user says "Rakesh", do not change it to "Rakesh Kumar"; if they say "19", do not change it to "28").
2. Only extract facts that are explicitly provided. Do not infer, guess, or invent details.
3. Return ONLY a valid JSON object mapping the lowercase keys (like "name", "age", "tagline", "profession", etc.) to their exact values as strings.
4. If no explicit factual details are provided in the prompt, return an empty JSON object: {}
5. Do not include markdown code fences or any other text before or after the JSON.
"""

def run_extractor(user_request: str) -> dict[str, str]:
    """
    Extracts explicit facts from the user request and returns them as a dictionary.
    """
    llm = _get_llm()

    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=f"User request: {user_request}"),
    ]

    try:
        response = llm.invoke(messages)
        if not isinstance(response.content, str):
            return {}
        
        raw = response.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())
        raw = raw.strip()

        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Ensure all values are strings
            return {str(k).lower(): str(v) for k, v in parsed.items()}
        return {}
    except Exception as e:
        print(f"[Extractor] Error during extraction: {e}")
        return {}
