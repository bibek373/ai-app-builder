"""
Validator Agent
===============
After the Coder generates files, this node inspects every ``.js`` file
using ``node --check`` (Node.js's built-in syntax checker).

Return value
------------
A list of dicts — one per failing file — that the Coder's fix pass will
consume:

    [{"filename": "app.js", "error": "SyntaxError: ..."}]

An empty list means all files passed validation (or there are no JS files /
Node.js is not installed, which counts as a pass for our purposes).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

# Ensure the project root is on sys.path so `state` is importable
# regardless of where Python is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv

from state import FileOutput

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "generated_projects")

# Maximum number of fix-and-revalidate cycles (must match graph.py constant)
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_validator(generated_files: list[FileOutput], app_name: str) -> list[dict]:
    """
    Validate all ``.js`` files in *generated_files* using ``node --check``.

    Parameters
    ----------
    generated_files : list[FileOutput]
        The files that the Coder just produced (in-memory list + on-disk).
    app_name : str
        The project folder name (used to locate files on disk).

    Returns
    -------
    list[dict]
        Each entry has ``{"filename": str, "error": str}`` for every file
        that failed validation.  Empty list = all clear.
    """
    # Graceful skip if Node.js is not available
    if not shutil.which("node"):
        print("[Validator] node not found on PATH — skipping JS validation.")
        return []

    safe_name = re.sub(r"[^\w\-]", "_", app_name.lower())
    js_files = [f for f in generated_files if f["filename"].endswith(".js")]

    if not js_files:
        print("[Validator] No .js files to validate.")
        return []

    print(f"\n[Validator] Checking {len(js_files)} JS file(s)...")
    errors: list[dict] = []

    for file_out in js_files:
        fname = file_out["filename"]
        abs_path = os.path.join(OUTPUT_DIR, safe_name, fname)

        if not os.path.exists(abs_path):
            print(f"[Validator] ⚠️  {fname} not found on disk — skipping.")
            continue

        try:
            result = subprocess.run(
                ["node", "--check", abs_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            msg = f"node --check timed out after 30 s"
            print(f"[Validator] ❌ {fname}: {msg}")
            errors.append({"filename": fname, "error": msg})
            continue
        except FileNotFoundError:
            # Race condition: node disappeared from PATH between the shutil.which
            # check and the subprocess call — bail out gracefully.
            print("[Validator] node disappeared from PATH — aborting validation.")
            break

        if result.returncode != 0:
            error_msg = (result.stderr or result.stdout or "Unknown syntax error").strip()
            print(f"[Validator] ❌ {fname}: {error_msg[:300]}")
            errors.append({"filename": fname, "error": error_msg})
        else:
            print(f"[Validator] ✅ {fname}: OK")

    return errors
