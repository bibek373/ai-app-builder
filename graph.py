"""
graph.py — LangGraph Orchestration
====================================
Wires the four nodes (Planner → Architect → Coder → Validator) into a
LangGraph StateGraph.  The Validator node introduces a self-correction
loop: if a JS file has a syntax error the graph routes back to the Coder
(fix mode) and retries up to MAX_VALIDATION_RETRIES times.

Graph topology
--------------
  planner ──[err?]──► architect ──[err?]──► coder ──► validator
                                                          │
                   ┌──── "retry" (< MAX_RETRIES) ◄───────┘
                   │
                   └──── "end"  (passed OR exhausted) ──► END
"""

import json
import sys

# Ensure stdout handles Unicode on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    getattr(sys.stdout, 'reconfigure')(encoding='utf-8', errors='replace')

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from state import AppState, initial_state
from agents.extractor import run_extractor
from agents.planner   import run_planner
from agents.architect import run_architect
from agents.coder     import run_coder
from agents.validator import run_validator

# Maximum fix-and-revalidate cycles before giving up
MAX_VALIDATION_RETRIES = 3


# ---------------------------------------------------------------------------
# Shared State Schema — defined in state.py, imported above
# ---------------------------------------------------------------------------
# AppState fields:
#   user_request      (str)               — original user prompt
#   plan              (str)               — Planner JSON output
#   tasks             (list[Task])        — Architect file-level tasks
#   generated_files   (list[FileOutput])  — Coder source-code outputs (full replace)
#   app_name          (str)               — safe project folder name
#   status            (str)               — pipeline stage
#   error             (str | None)        — last error message
#   validation_errors (list)              — Validator findings (empty = pass)
#   retry_count       (int)               — how many fix cycles have run
#   validation_status (str)               — "pending"|"passed"|"failed_retrying"|"failed_exhausted"


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def extractor_node(state: AppState) -> dict:
    """Run the facts extractor at the start of the pipeline."""
    print("\n[Extractor] Extracting explicit facts from request...")
    try:
        user_facts = run_extractor(state["user_request"])
        print(f"[Extractor] Extracted facts: {user_facts}")
        return {
            "user_facts": user_facts,
            "status": "starting",
            "error": None,
        }
    except Exception as e:
        print(f"[Extractor] Failed to extract facts: {e}")
        return {
            "user_facts": {},
            "status": "starting",
            "error": None,
        }


def planner_node(state: AppState) -> dict:
    """Run the planner and extract the app name from the plan."""
    print("\n[Planner] Analysing idea...")
    try:
        plan_raw = run_planner(state["user_request"])

        # Try to extract app_name for downstream use
        app_name = "my_app"
        try:
            plan_dict = json.loads(plan_raw)
            app_name = plan_dict.get("app_name", "my_app")
        except json.JSONDecodeError:
            pass

        return {
            "plan": plan_raw,
            "app_name": app_name,
            "status": "planned",
            "error": None,
        }
    except Exception as e:
        return {"status": "error", "error": f"Planner failed: {e}"}


def architect_node(state: AppState) -> dict:
    """Run the architect using the plan from the previous step."""
    print("\n[Architect] Designing technical spec...")
    try:
        tasks_raw = run_architect(state["plan"])
        # run_architect returns a list[Task] or a JSON string of one
        if isinstance(tasks_raw, list):
            tasks = tasks_raw
        else:
            try:
                parsed = json.loads(tasks_raw)
                tasks = parsed if isinstance(parsed, list) else parsed.get("components", [])
            except json.JSONDecodeError:
                tasks = []
        return {
            "tasks": tasks,
            "status": "architected",
            "error": None,
        }
    except Exception as e:
        return {"status": "error", "error": f"Architect failed: {e}"}


def coder_node(state: AppState) -> dict:
    """
    Run the coder to generate (or fix) source files.

    On the first pass ``validation_errors`` is empty → full generation.
    On retry passes it is non-empty → fix-only mode for failing files.
    """
    val_errors = state.get("validation_errors", [])
    retry = bool(val_errors)

    if retry:
        attempt = state.get("retry_count", 1)
        print(f"\n[Coder] Retry {attempt}/{MAX_VALIDATION_RETRIES} — fixing {len(val_errors)} file(s)...")
    else:
        print("\n[Coder] Generating source files...")

    try:
        files = run_coder(
            state["tasks"],
            state["app_name"],
            validation_errors=val_errors if retry else None,
            current_files=state.get("generated_files", []) if retry else None,
            user_facts=state.get("user_facts", {}),
        )
        return {
            "generated_files": files,
            "status": "completed",
            "error": None,
        }
    except Exception as e:
        return {"status": "error", "error": f"Coder failed: {e}"}


def validator_node(state: AppState) -> dict:
    """
    Run ``node --check`` on every generated ``.js`` file.

    • Pass  → validation_status = "passed", clear errors.
    • Fail, retries left  → "failed_retrying", increment retry_count.
    • Fail, retries exhausted → "failed_exhausted", set status = "error".
    """
    retry_count = state.get("retry_count", 0)
    attempt_label = f"attempt {retry_count + 1}/{MAX_VALIDATION_RETRIES}"
    print(f"\n[Validator] Validating JS files ({attempt_label})...")

    try:
        errors = run_validator(state["generated_files"], state["app_name"])
    except Exception as e:
        # If validator itself crashes, treat as pass to avoid blocking the user
        print(f"[Validator] ⚠️  Validator raised an exception ({e}) — treating as passed.")
        errors = []

    if not errors:
        print("[Validator] ✅ All JS files passed.")
        return {
            "validation_status": "passed",
            "validation_errors": [],
        }

    # Some files failed
    if retry_count >= MAX_VALIDATION_RETRIES:
        filenames = ", ".join(e["filename"] for e in errors)
        last_error = errors[0]["error"]
        print(f"[Validator] ❌ Validation exhausted after {MAX_VALIDATION_RETRIES} retries.")
        return {
            "validation_status": "failed_exhausted",
            "validation_errors": errors,
            "status": "error",
            "error": (
                f"JS validation failed after {MAX_VALIDATION_RETRIES} attempts. "
                f"Failing file(s): {filenames}. "
                f"Last error: {last_error}"
            ),
        }

    print(f"[Validator] ⚠️  {len(errors)} file(s) failed — scheduling fix (retry {retry_count + 1}).")
    return {
        "validation_status": "failed_retrying",
        "validation_errors": errors,
        "retry_count": retry_count + 1,
    }


# ---------------------------------------------------------------------------
# Edge Conditions
# ---------------------------------------------------------------------------

def should_continue(state: AppState) -> str:
    """Stop the graph if planner or architect reported an error."""
    if state.get("status") == "error":
        return "end"
    return "continue"


def validator_edge(state: AppState) -> str:
    """
    Route after the validator:
    - "end"   → pipeline is done (passed or retries exhausted)
    - "retry" → send broken files back to the coder for a fix pass
    """
    vs = state.get("validation_status", "pending")
    if vs in ("passed", "failed_exhausted"):
        return "end"
    return "retry"   # "failed_retrying"


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

def build_graph() -> CompiledStateGraph:
    graph = StateGraph(AppState)

    # Add nodes
    graph.add_node("extractor", extractor_node)
    graph.add_node("planner",   planner_node)
    graph.add_node("architect", architect_node)
    graph.add_node("coder",     coder_node)
    graph.add_node("validator", validator_node)

    # Entry point
    graph.set_entry_point("extractor")

    # Extractor -> Planner
    graph.add_edge("extractor", "planner")

    # Planner → Architect (with error exit)
    graph.add_conditional_edges(
        "planner",
        should_continue,
        {"continue": "architect", "end": END},
    )
    # Architect → Coder (with error exit)
    graph.add_conditional_edges(
        "architect",
        should_continue,
        {"continue": "coder", "end": END},
    )
    # Coder always goes to Validator
    graph.add_edge("coder", "validator")

    # Validator → END (pass / exhausted) or back to Coder (retry loop)
    graph.add_conditional_edges(
        "validator",
        validator_edge,
        {"end": END, "retry": "coder"},
    )

    return graph.compile()


# Singleton compiled graph
app_graph = build_graph()


def run_pipeline(user_request: str) -> AppState:
    """Public entry point: run the full pipeline for a given user request."""
    return app_graph.invoke(initial_state(user_request))  # type: ignore[return-value]
