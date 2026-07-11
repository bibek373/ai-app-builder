"""
Coder Agent
===========
Receives the list of Tasks from the Architect and generates production-
quality source code for each file.

For every Task it:
  1. Calls the LLM with the task's description + implementation_notes.
  2. Wraps the response in a FileOutput TypedDict.
  3. Writes the file to  generated_projects/<app_name>/<filename>.
  4. Returns the list of FileOutput dicts → AppState["generated_files"].

In retry mode (when validation_errors is supplied by the Validator node),
only the files that failed validation are re-generated; working files are
returned unchanged.  The fixed file(s) are overwritten on disk.
"""

from __future__ import annotations

import os
import re
import sys
import time

# Ensure the project root is on sys.path so `state` is importable
# regardless of where Python is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from state import Task, FileOutput

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "generated_projects")

# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

_EXT_TO_LANG: dict[str, str] = {
    ".py":    "python",
    ".js":    "javascript",
    ".ts":    "typescript",
    ".jsx":   "jsx",
    ".tsx":   "tsx",
    ".html":  "html",
    ".css":   "css",
    ".json":  "json",
    ".md":    "markdown",
    ".yaml":  "yaml",
    ".yml":   "yaml",
    ".sh":    "bash",
    ".toml":  "toml",
    ".txt":   "text",
    ".env":   "text",
    ".sql":   "sql",
}

def _detect_language(filename: str) -> str:
    ext = os.path.splitext(filename)[-1].lower()
    return _EXT_TO_LANG.get(ext, "text")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGroq:
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),  # type: ignore[arg-type]
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0.1,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CODER_SYSTEM_PROMPT = """\
You are an expert software engineer and web designer writing production-quality code.

GENERAL RULES:
1. Output ONLY the raw file content — no markdown fences, no prose.
2. The very first character of your response must be the first character
   of the file (e.g. a shebang, import statement, or HTML doctype).
3. Use modern best practices for the language / framework in use.
4. Add helpful inline comments, but keep them concise.
5. Handle edge cases and errors gracefully.
6. The code must be complete and immediately runnable — no TODOs,
   no placeholder functions, no "add your logic here" stubs.

ZERO-PLACEHOLDER RULE (CRITICAL):
7. NEVER include placeholder text, bracketed instructions, or template
   markers in the output. Specifically FORBIDDEN patterns:
   - "[Insert ...]", "[Add ...]", "[Your ...]", "[Placeholder ...]"
   - "Lorem ipsum" dummy text
   - "TODO:", "FIXME:", "add your logic here"
   - Descriptions of what SHOULD go in a spot instead of actual content
   Every piece of text in the generated file must be real, meaningful,
   finished content appropriate to the project's purpose. If you don't
   know a specific detail, write realistic, relevant content that fits
   the context — never leave an instruction or placeholder for the user
   to fill in.

HTML / STATIC-SITE RULES:
8. For HTML files, always reference stylesheets (CSS) and scripts (JS)
   using relative paths (e.g. 'styles.css', './styles.css',
   'css/styles.css' — never starting with a leading slash '/').
9. Ensure the name and path of the linked stylesheet in index.html
   matches the actual generated CSS filename exactly.

DESIGN-QUALITY RULES FOR STATIC WEBSITES:
10. SECTION SEPARATION — Do NOT use a single gradient/color for the
    entire page.  Alternate background colors or shades between
    sections (e.g. white → light-gray → white → accent-tint) so each
    section is visually distinct.  You may also use card or container
    styling within sections.
11. TYPOGRAPHY HIERARCHY — Use a curated Google Fonts import (e.g. Inter,
    Outfit, Poppins).  Headings should be large and bold with generous
    margin-bottom.  Body text should NEVER run edge-to-edge; wrap all
    section content in a container with max-width (e.g. 1200px) and
    margin: 0 auto; padding: 0 2rem.
12. CONSISTENT SPACING — Every section should have generous vertical
    padding (e.g. padding: 5rem 0;) so content never feels cramped.
    Maintain uniform spacing between all elements.
13. STYLED CONTENT BLOCKS — Never render information as plain <ul>/<li>
    bullet lists.  Instead use styled cards, icon+text grid tiles,
    or feature boxes with background, shadow, border-radius, and
    hover effects.  Use CSS Grid or Flexbox for multi-column layouts.
14. LIGHTWEIGHT CSS-ONLY SCROLL ANIMATIONS — Add fade-in / slide-up
    reveal effects on sections as the user scrolls.  Implement using:
    a) A CSS class `.reveal { opacity: 0; transform: translateY(30px);
       transition: opacity 0.8s ease, transform 0.8s ease; }`
    b) A CSS class `.reveal.active { opacity: 1; transform: translateY(0); }`
    c) A small vanilla JS snippet using IntersectionObserver to add
       the `.active` class when each `.reveal` element enters the
       viewport.  No heavy libraries.
15. VISUAL POLISH — Add subtle box-shadows, rounded corners, hover
    scale/lift effects on cards and buttons, smooth transitions
    (transition: all 0.3s ease), and at least one gradient accent
    (e.g. hero background, CTA button).
16. HERO SECTION — Always include a visually prominent hero with a
    gradient or image background, large heading, descriptive subtitle
    (with REAL text, not a placeholder), and a CTA button if
    appropriate.
17. FOOTER — Include a simple, styled footer with background color
    contrast.
18. Keep it lightweight — NO backend, NO frameworks, NO heavy JS
    libraries.  Pure HTML + CSS + vanilla JS only.

FACT-PRESERVATION RULE:
19. CRITICAL: For any explicit user-provided facts (such as name, age,
    tagline), you MUST wrap the exact value inside the HTML with a
    `data-fact="<key>"` attribute in its wrapping tag (e.g.
    `<h1 data-fact="name">...</h1>`, `<p data-fact="age">...</p>`).
    Post-processing relies on the `data-fact` attribute to force the
    exact user facts in the final page, so the attribute MUST be
    present on the tags enclosing those facts.
"""

FIX_SYSTEM_PROMPT = """\
You are an expert software engineer fixing a syntax error in a generated file.

RULES:
1. Output ONLY the corrected raw file content — no markdown fences, no prose.
2. Fix ONLY the reported syntax error(s). Do not refactor or otherwise change
   code that was already working.
3. The returned file must be complete and immediately runnable.
4. The very first character of your response must be the first character
   of the file.
"""

# ---------------------------------------------------------------------------
# Retry helper for Groq rate-limit (429) errors
# ---------------------------------------------------------------------------

_MAX_LLM_RETRIES = 4
_BASE_WAIT_SECS  = 4  # first retry waits 4 s, then 8, 16, 32


def _call_with_retry(llm, messages, context_label: str = "") -> str:
    """Invoke the LLM with automatic retry + exponential back-off on 429."""
    for attempt in range(1, _MAX_LLM_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            if not isinstance(response.content, str):
                raise TypeError(
                    f"Expected LLM response to be a string, got {type(response.content)}"
                )
            code = response.content.strip()
            # Strip accidental markdown fences
            code = re.sub(r"^```[a-zA-Z]*\n?", "", code)
            code = re.sub(r"\n?```$", "", code.strip())
            return code.strip()
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < _MAX_LLM_RETRIES:
                wait = _BASE_WAIT_SECS * (2 ** (attempt - 1))
                print(
                    f"  [Rate-limit] {context_label} attempt {attempt}/{_MAX_LLM_RETRIES} "
                    f"hit 429 — retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                raise  # non-429 or final attempt — propagate
    # Should never reach here, but satisfy type checkers
    raise RuntimeError("Exhausted retries")  # pragma: no cover


# ---------------------------------------------------------------------------
# Single-file code generation
# ---------------------------------------------------------------------------

def _generate_code(task: Task) -> str:
    """Call the LLM to generate source code for one Task."""
    llm = _get_llm()

    prompt = (
        f"File to generate: {task['filename']}\n\n"
        f"Purpose:\n{task['description']}\n\n"
        f"Dependencies: {', '.join(task['dependencies']) or 'none'}\n\n"
        f"Implementation instructions:\n{task['implementation_notes']}\n\n"
        "Generate the complete file content now."
    )

    messages = [
        SystemMessage(content=CODER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    return _call_with_retry(llm, messages, context_label=task['filename'])


# ---------------------------------------------------------------------------
# Write helper
# ---------------------------------------------------------------------------

def _write_file(app_name: str, filename: str, content: str) -> str:
    """
    Write content to  generated_projects/<safe_app_name>/<filename>.
    Creates any intermediate directories (handles both flat files and
    nested paths like src/components/App.js).
    Returns the absolute path of the written file.
    """
    safe_name = re.sub(r"[^\w\-]", "_", app_name.lower())
    dest = os.path.join(OUTPUT_DIR, safe_name, filename)
    if filename.endswith('/') or filename.endswith('\\') or os.path.isdir(dest):
        os.makedirs(dest, exist_ok=True)
        return dest
    # Always create the full directory tree leading to dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(content)
    return dest


# ---------------------------------------------------------------------------
# Error-guided fix helper (used in retry mode)
# ---------------------------------------------------------------------------

def _fix_code(task: Task, current_content: str, error_msg: str) -> str:
    """Re-generate a single file, giving the LLM the current broken content
    and the exact ``node --check`` error so it can make a targeted fix."""
    llm = _get_llm()

    prompt = (
        f"File: {task['filename']}\n\n"
        f"Current file content (contains a syntax error):\n"
        f"```\n{current_content}\n```\n\n"
        f"Syntax error reported by node --check:\n{error_msg}\n\n"
        "Return the complete corrected file content now."
    )

    messages = [
        SystemMessage(content=FIX_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    return _call_with_retry(llm, messages, context_label=f"fix:{task['filename']}")


# ---------------------------------------------------------------------------
# Post-processing: strip any remaining placeholder / instruction text
# ---------------------------------------------------------------------------

# Matches patterns like [Insert ...], [Add ...], [Your ...], [Placeholder ...]
_PLACEHOLDER_RE = re.compile(
    r'\['
    r'(?:Insert|Add|Your|Placeholder|Enter|Replace|Put|Write|Fill|Update|Include|Provide)'
    r'\s[^\]]{3,}'
    r'\]',
    re.IGNORECASE,
)


def _strip_placeholders(content: str, filename: str) -> str:
    """Remove any bracketed placeholder / instruction text that the LLM
    accidentally left in the generated file.  Only applied to HTML files."""
    if not filename.lower().endswith((".html", ".htm")):
        return content

    cleaned = _PLACEHOLDER_RE.sub("", content)
    if cleaned != content:
        # Count how many were removed for logging
        count = len(_PLACEHOLDER_RE.findall(content))
        print(f"  [PostProc] Removed {count} placeholder(s) from {filename}")
    return cleaned


def _apply_user_facts(html_content: str, user_facts: dict[str, str] | None) -> str:
    """
    Finds elements with attribute data-fact="<key>" or id="<key>" in HTML and replaces
    their inner content with the exact value from user_facts.
    """
    if not user_facts:
        return html_content

    processed = html_content
    for key, value in user_facts.items():
        # Match pattern: (<[a-zA-Z0-9]+[^>]*\b(data-fact|id)=["\']key["\'][^>]*>)(.*?)(</[a-zA-Z0-9]+>)
        # Group 1: start tag
        # Group 2: data-fact or id
        # Group 3: text value
        # Group 4: closing tag
        pattern = re.compile(
            rf'(<[a-zA-Z0-9]+[^>]*\b(data-fact|id)=["\']{re.escape(key)}["\'][^>]*>)(.*?)(</[a-zA-Z0-9]+>)',
            re.DOTALL | re.IGNORECASE
        )
        processed = pattern.sub(rf'\g<1>{value}\g<4>', processed)
    return processed


def _fix_failing_files(
    tasks: list[Task],
    app_name: str,
    validation_errors: list,
    current_files: list[FileOutput],
    user_facts: dict[str, str] | None = None,
) -> list[FileOutput]:
    """Fix only the files listed in *validation_errors*, leave the rest intact.

    Returns the complete updated ``generated_files`` list (all files, with
    broken ones replaced by their corrected versions).
    """
    # Build fast-lookup maps
    error_map:   dict[str, str]        = {ve["filename"]: ve["error"] for ve in validation_errors}
    content_map: dict[str, str]        = {f["filename"]: f["content"] for f in current_files}
    task_map:    dict[str, Task]       = {t["filename"]: t for t in tasks}

    updated = list(current_files)  # shallow copy — we replace entries in place

    for i, file_out in enumerate(updated):
        fname = file_out["filename"]
        if fname not in error_map:
            continue  # file is fine, skip

        print(f"  [Coder/Fix] Fixing {fname} ...")

        task = task_map.get(fname)
        if task is None:
            print(f"  [!] No task found for {fname} — cannot fix, leaving as-is.")
            continue

        try:
            fixed_content = _fix_code(task, content_map.get(fname, ""), error_map[fname])
            fixed_content = _strip_placeholders(fixed_content, fname)
        except Exception as exc:
            print(f"  [!] Fix LLM call failed for {fname}: {exc}")
            continue

        if fname.endswith(".html") and user_facts:
            fixed_content = _apply_user_facts(fixed_content, user_facts)

        abs_path = _write_file(app_name, fname, fixed_content)
        print(f"  [OK] Fixed → {abs_path}")

        updated[i] = FileOutput(
            filename=fname,
            content=fixed_content,
            language=_detect_language(fname),
        )

    return updated


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_coder(
    tasks: list[Task],
    app_name: str,
    *,
    validation_errors: list | None = None,
    current_files: list[FileOutput] | None = None,
    user_facts: dict[str, str] | None = None,
) -> list[FileOutput]:
    """
    Generate (or fix) source files and write them to disk.

    Parameters
    ----------
    tasks : list[Task]
        File-level tasks produced by the Architect.
    app_name : str
        Project sub-directory name under ``generated_projects/``.
    validation_errors : list | None
        If supplied (non-empty), the coder runs in *fix mode*: only the
        files listed here are re-generated; all other files are passed
        through unchanged from *current_files*.
    current_files : list[FileOutput] | None
        The existing generated-file list.  Required when *validation_errors*
        is provided.
    user_facts : dict[str, str] | None
        Explicit factual details provided by the user to inject in HTML.
    """
    # ── Retry / fix mode ──────────────────────────────────────────────────
    if validation_errors:
        print(f"\n[Coder] Fix mode — repairing {len(validation_errors)} file(s)...")
        return _fix_failing_files(tasks, app_name, validation_errors, current_files or [], user_facts)

    # ── Initial generation mode ────────────────────────────────────────────
    generated: list[FileOutput] = []

    for task in tasks:
        fname = task["filename"]
        # If it's a directory, create it and skip LLM generation
        if fname.endswith('/') or fname.endswith('\\'):
            abs_path = _write_file(app_name, fname, "")
            print(f"  [OK] Created directory → {abs_path}")
            generated.append(
                FileOutput(
                    filename=fname,
                    content="",
                    language="text",
                )
            )
            continue

        print(f"  [Coder] Coding: {fname} ...")

        try:
            content = _generate_code(task)
            content = _strip_placeholders(content, fname)
        except Exception as exc:
            print(f"  [!] Failed to generate {fname}: {exc}")
            content = f"# Generation failed: {exc}\n"

        if fname.endswith(".html") and user_facts:
            content = _apply_user_facts(content, user_facts)

        # Write to disk
        abs_path = _write_file(app_name, fname, content)
        print(f"  [OK] Written → {abs_path}")

        generated.append(
            FileOutput(
                filename=fname,
                content=content,
                language=_detect_language(fname),
            )
        )

    return generated
