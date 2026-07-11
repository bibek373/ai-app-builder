"""
state.py — Shared Pipeline State
==================================
Defines the single TypedDict that flows through every node in the
LangGraph pipeline.  Each agent receives this state, reads what it
needs, and returns a partial dict with only the fields it updates.

Field ownership:
  user_request      → set once by the caller (never mutated by agents)
  plan              → written by the Planner node
  tasks             → written by the Architect node
  generated_files   → written (fully replaced) by Coder node each pass
  validation_errors → written by Validator node; consumed by Coder fix pass
  retry_count       → incremented by Validator node on each failed attempt
  validation_status → written by Validator node
"""

from __future__ import annotations

from typing import Annotated, TypedDict
import operator


class FileOutput(TypedDict):
    """
    Represents a single generated source file.

    Attributes
    ----------
    filename : str
        Relative path of the file inside the generated project directory.
        Examples: "app.py", "src/components/Header.jsx"
    content : str
        The full, ready-to-write source code for this file.
    language : str
        Programming language / file type (e.g. "python", "javascript",
        "html", "css", "markdown").  Used for syntax highlighting in the UI.
    """

    filename: str
    content: str
    language: str


class Task(TypedDict):
    """
    Represents a single file-level task produced by the Architect.

    Attributes
    ----------
    filename : str
        The target file path for this task (mirrors FileOutput.filename).
    description : str
        Plain-English summary of what the file should contain / do.
    dependencies : list[str]
        Other filenames or packages this file depends on.
    implementation_notes : str
        Detailed guidance for the Coder on how to implement this file.
    """

    filename: str
    description: str
    dependencies: list[str]
    implementation_notes: str


class ValidationError(TypedDict):
    """
    A syntax error found in a generated file by the Validator node.

    Attributes
    ----------
    filename : str
        The relative filename of the file that failed validation.
    error : str
        The raw error output from ``node --check`` (or equivalent).
    """

    filename: str
    error: str


class AppState(TypedDict):
    """
    The single shared state object that travels through the entire
    LangGraph pipeline.

    Fields
    ------
    user_request : str
        The original plain-English prompt entered by the user.
        Set once at pipeline startup; never overwritten by any agent.

    plan : str
        High-level project plan produced by the Planner agent.
        Includes app name, tech stack, feature list, and overall approach.
        Stored as a raw string (typically JSON) so it can be inspected or
        re-parsed downstream.

    tasks : list[Task]
        File-level work items produced by the Architect agent.
        Each Task tells the Coder exactly what to write for one file.
        Annotated with operator.add so LangGraph can merge partial updates
        from parallel coder fan-outs in future phases.

    generated_files : list[FileOutput]
        The actual source-code outputs produced by the Coder agent.
        Each entry corresponds to one Task and contains the full file
        content ready to be written to disk.
        NOT annotated with operator.add — each Coder pass fully replaces
        the list so the retry loop can update individual files cleanly.

    app_name : str
        Short, filesystem-safe name for the project.
        Extracted from the plan by the Planner and used to create the
        output directory under generated_projects/.

    status : str
        Current pipeline stage.  One of:
          "starting" | "planned" | "architected" | "coding" |
          "completed" | "error"

    error : str | None
        Human-readable error message if any node fails; None otherwise.

    validation_errors : list[ValidationError]
        Syntax errors found by the Validator node.  Empty list means all
        files passed.  Populated on each failed pass; reset to [] by the
        Coder node at the start of every fix pass.

    retry_count : int
        Number of validation-fix cycles attempted so far.  The Validator
        increments this on each failed pass; capped at MAX_VALIDATION_RETRIES.

    validation_status : str
        One of: "pending" | "passed" | "failed_retrying" | "failed_exhausted"
    """

    # ── Input ──────────────────────────────────────────────────────────────
    user_request: str
    user_facts: dict[str, str]

    # ── Planner output ─────────────────────────────────────────────────────
    plan: str

    # ── Architect output ───────────────────────────────────────────────────
    tasks: Annotated[list[Task], operator.add]

    # ── Coder output ───────────────────────────────────────────────────────
    generated_files: list[FileOutput]  # plain list — replaced on each pass

    # ── Pipeline metadata ──────────────────────────────────────────────────
    app_name: str
    status: str
    error: str | None

    # ── Validator / self-correction loop ───────────────────────────────────
    validation_errors: list  # list[ValidationError]
    retry_count: int
    validation_status: str   # "pending"|"passed"|"failed_retrying"|"failed_exhausted"


# ---------------------------------------------------------------------------
# Helper — build a clean initial state from a user prompt
# ---------------------------------------------------------------------------

def initial_state(user_request: str) -> AppState:
    """
    Returns a fresh AppState populated with the user's request and safe
    defaults for every other field.  Pass this to ``app_graph.invoke()``.

    Parameters
    ----------
    user_request : str
        The raw prompt from the user.

    Returns
    -------
    AppState
    """
    return AppState(
        user_request=user_request,
        user_facts={},
        plan="",
        tasks=[],
        generated_files=[],
        app_name="my_app",
        status="starting",
        error=None,
        validation_errors=[],
        retry_count=0,
        validation_status="pending",
    )
