"""
Architect Agent
===============
Receives the planner's JSON plan string and expands it into a list of
file-level Task dicts (AppState["tasks"]).

Each Task tells the Coder exactly what to write for one file:
  - filename             : relative path (mirrors file_structure from plan)
  - description          : plain-English purpose of the file
  - dependencies         : other files or pip packages this file needs
  - implementation_notes : detailed, step-by-step coding instructions

Output JSON schema (returned as a JSON array)
---------------------------------------------
[
  {
    "filename":             "relative/path/file.py",
    "description":          "what this file does",
    "dependencies":         ["other_file.py", "some-package"],
    "implementation_notes": "detailed guidance for the coder"
  },
  ...
]
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

from state import Task

load_dotenv()

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),  # type: ignore[arg-type]
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.2,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ARCHITECT_SYSTEM_PROMPT = """\
You are a senior software architect.
Given a high-level project plan (JSON), break it down into a list of
file-level implementation tasks for a code-generation agent.

CORE RULES:
1. Reply with ONLY a valid JSON array — no markdown fences, no prose.
2. One object per file listed in the plan's "file_structure".
3. "implementation_notes" must be detailed enough that a junior developer
   could implement the file without asking questions.
4. List actual pip packages in "dependencies", not vague terms.
5. For web projects (HTML/CSS/JS), the implementation instructions must
   explicitly specify that stylesheet (CSS) and script (JS) paths linked
   in HTML files must be relative (e.g., 'styles.css' or 'css/styles.css',
   never starting with a leading slash '/'), ensuring they resolve correctly
   when index.html is opened directly via file:// in a browser.  Ensure the
   CSS filename in the task list matches the HTML link path exactly.

ZERO-PLACEHOLDER RULE (CRITICAL):
6. The implementation_notes must instruct the coder to write COMPLETE,
   REAL content for every text element.  NEVER include bracketed
   placeholders like "[Insert ...]", "[Your ...]", "[Add ...]" in
   descriptions or notes.  If the plan lacks a specific detail, tell
   the coder to write realistic, relevant copy — never to leave a
   placeholder for the user to fill in.

FACT-PRESERVATION RULE:
7. If the input plan contains specific factual details (like specific
   names, ages, taglines, or numbers), you MUST pass these details
   EXACTLY as given into the task descriptions and implementation notes.
   Do NOT paraphrase, generalize, or change them.
8. CRITICAL: When styling or structuring HTML pages, the tasks must
   instruct the coder to wrap explicit facts in HTML tags with a
   `data-fact="<key>"` attribute (e.g., `<h1 data-fact="name">Rakesh</h1>`).
   This is required for post-processing injection.

DESIGN-QUALITY RULES FOR STATIC WEBSITES:
9.  In implementation_notes for HTML files, ALWAYS specify:
    - Alternating section backgrounds (e.g. white / light-gray / accent-tint)
      so sections are visually distinct — never one gradient for the whole page.
    - A centered content container (max-width ~1200px, margin: 0 auto,
      padding: 0 2rem) inside each section.
    - Generous vertical section padding (e.g. padding: 5rem 0).
10. In implementation_notes for CSS files, ALWAYS specify:
    - A Google Fonts import (e.g. Inter, Poppins, Outfit).
    - Styled cards/grid tiles instead of plain bullet lists for any
      feature or benefit lists — with background, shadow, border-radius,
      and hover effects.
    - A `.reveal` / `.reveal.active` CSS pattern for scroll-triggered
      fade-in / slide-up animations.
    - Subtle box-shadows, rounded corners, hover scale/lift effects on
      interactive elements, and smooth transitions.
11. In implementation_notes for JS files (if any), specify a lightweight
    IntersectionObserver snippet that adds `.active` to `.reveal` elements
    on scroll.  No heavy libraries.

Each array element MUST have exactly these keys:
{
  "filename":             "<relative file path matching the plan>",
  "description":          "<1-2 sentence purpose of this file>",
  "dependencies":         ["<file or package>", ...],
  "implementation_notes": "<detailed, numbered implementation steps>"
}
"""

# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_architect(plan: str) -> list[Task]:
    """
    Takes the planner's JSON plan string and returns a list of Task dicts.
    This is what gets stored in AppState["tasks"].
    """
    llm = _get_llm()

    messages = [
        SystemMessage(content=ARCHITECT_SYSTEM_PROMPT),
        HumanMessage(content=f"Project plan:\n{plan}"),
    ]

    response = llm.invoke(messages)
    if not isinstance(response.content, str):
        raise TypeError(f"Expected LLM response to be a string, got {type(response.content)}")
    raw = response.content.strip()

    # Strip markdown fences
    raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw.strip())
    raw = raw.strip()

    # Parse — expect a JSON array
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Architect returned invalid JSON.\nRaw output:\n{raw}"
        ) from exc

    # Normalise: accept either a bare array or {"components": [...]}
    if isinstance(parsed, list):
        tasks_raw = parsed
    elif isinstance(parsed, dict):
        # Try common wrapper keys
        for key in ("components", "tasks", "files"):
            if key in parsed and isinstance(parsed[key], list):
                tasks_raw = parsed[key]
                break
        else:
            raise ValueError(f"Architect JSON has no recognisable list key: {list(parsed.keys())}")
    else:
        raise ValueError(f"Architect returned unexpected type: {type(parsed)}")

    # Coerce each item into a Task TypedDict
    tasks: list[Task] = []
    for item in tasks_raw:
        tasks.append(
            Task(
                filename=item.get("filename", item.get("file", "unknown.txt")),
                description=item.get("description", item.get("purpose", "")),
                dependencies=item.get("dependencies", []),
                implementation_notes=item.get("implementation_notes", ""),
            )
        )

    return tasks
