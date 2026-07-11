"""
Planner Agent
=============
Receives the user's plain-English request and produces a structured
high-level plan (stored as a JSON string in AppState["plan"]).

Output JSON schema
------------------
{
  "app_name":            "filesystem-safe project name (snake_case)",
  "description":         "one-paragraph description of the app",
  "tech_stack":          ["list", "of", "languages / frameworks / tools"],
  "features":            ["core feature 1", "feature 2", ...],
  "file_structure":      ["relative/path/file1.py", "file2.html", ...],
  "special_requirements":"any extra constraints, or null"
}
"""

from __future__ import annotations

import json
import os
import re
import sys

# Ensure the project root is on sys.path so `state` is importable
# regardless of where Python is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),  # type: ignore[arg-type]
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are an expert software project planner.
Given a user's app idea, produce a concise, actionable plan.

CRITICAL FACT-PRESERVATION RULE:
If the user provides specific factual details (like specific names, ages, numbers, statuses, titles, or taglines), you MUST preserve them EXACTLY as given in all fields of the plan (e.g. description, features list). Do NOT paraphrase, generalize, omit, invent, or "improve" factual details (e.g. if the user says name "Rakesh", do not change it to "Rakesh Kumar"; if they say age "19", do not change it to "28"; if they say tagline "Aspiring Government Officer", do not change or expand it). Only fill in placeholder content for details the user did NOT specify.

RULES:
1. Reply with ONLY valid JSON — no markdown fences, no prose before or after.
2. Use snake_case for "app_name" (e.g. "todo_app", "weather_dashboard").
3. Keep "file_structure" realistic — only files that will actually be coded.
4. Limit scope to what can be generated in one pass (≤ 10 files).
5. For web projects (HTML/CSS/JS), always name the stylesheet file 'styles.css' (or placed in a subdirectory like 'css/styles.css'). All paths in "file_structure" must be relative.
6. If the user provides specific factual details (names, ages, numbers, statuses), you MUST preserve them EXACTLY as given. Do NOT paraphrase, invent, or 'improve' factual details. Only fill in reasonable placeholder content for details the user did NOT specify.

Required JSON keys:
{
  "app_name":            "<snake_case string>",
  "description":         "<1-2 sentence summary>",
  "tech_stack":          ["<language/framework>", ...],
  "features":            ["<feature>", ...],
  "file_structure":      ["<relative file path>", ...],
  "special_requirements":"<string or null>"
}
"""

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_planner(user_request: str) -> str:
    """
    Calls the LLM with the user's request and returns the plan as a
    raw JSON string (AppState["plan"]).

    The returned string is validated to be parseable JSON; if the LLM
    wraps it in markdown fences those are stripped first.
    """
    llm = _get_llm()

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"App idea: {user_request}"),
    ]

    response = llm.invoke(messages)
    if not isinstance(response.content, str):
        raise TypeError(f"Expected LLM response to be a string, got {type(response.content)}")
    raw = response.content.strip()

    # Strip markdown code fences if the LLM added them
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())
    raw = raw.strip()

    # Validate — raises ValueError if LLM returned garbage
    try:
        json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Planner returned invalid JSON.\nRaw output:\n{raw}"
        ) from exc

    return raw
