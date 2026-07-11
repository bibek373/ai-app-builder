"""
app.py — Streamlit Frontend
============================
The main user-facing interface for the AI App Builder.
Users describe their app idea; the pipeline generates the full project.
"""

import streamlit as st
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from graph import run_pipeline

load_dotenv()

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI App Builder",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inline CSS — premium dark UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Header */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.25rem;
}
.hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.badge-success { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid #34d399; }
.badge-running { background: rgba(96,165,250,0.2); color: #60a5fa; border: 1px solid #60a5fa; }
.badge-error   { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid #f87171; }

/* File tree */
.file-item {
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    color: #a78bfa;
    padding: 0.2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    api_key_input = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Your Groq API key (or set GROQ_API_KEY in .env)",
    )

    model_choice = st.selectbox(
        "Model",
        [
            "openai/gpt-oss-120b",   # High quality — recommended
            "openai/gpt-oss-20b",    # Faster / cheaper
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ],
        help="Choose the Groq model to use for generation",
    )

    st.markdown("---")
    st.markdown("### 📂 Generated Projects")

    output_dir = Path(os.getenv("OUTPUT_DIR", "generated_projects"))
    if output_dir.exists():
        projects = [d for d in output_dir.iterdir() if d.is_dir()]
        if projects:
            for proj in sorted(projects):
                st.markdown(f"📁 `{proj.name}`")
        else:
            st.caption("No projects generated yet.")
    else:
        st.caption("Output directory will appear here.")

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Built with LangGraph + Groq + Streamlit</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">🚀 AI App Builder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Describe your app idea — the AI agents will plan, architect, and code it for you.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main Input Area
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    user_request = st.text_area(
        label="Your App Idea",
        placeholder=(
            "e.g. A task management web app with user authentication, "
            "drag-and-drop kanban board, and a REST API backend using FastAPI and SQLite..."
        ),
        height=140,
        label_visibility="collapsed",
    )

    generate_btn = st.button(
        "✨ Generate My App",
        use_container_width=True,
        type="primary",
        disabled=not user_request.strip(),
    )

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------
if generate_btn and user_request.strip():
    # Inject API key at runtime if provided
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
    if model_choice:
        os.environ["GROQ_MODEL"] = model_choice

    st.markdown("---")
    st.markdown("### 🔄 Pipeline Running")

    prog_bar = st.progress(0, text="Starting pipeline...")

    with st.spinner("Agents are collaborating..."):
        try:
            prog_bar.progress(10, text="🧠 Planner: Analysing your idea...")

            result = run_pipeline(user_request)

            prog_bar.progress(100, text="✅ Pipeline complete!")

            if result.get("status") == "error":
                st.error(f"❌ Pipeline error: {result.get('error')}")

                # Surface validation-exhausted details prominently
                if result.get("validation_status") == "failed_exhausted":
                    st.markdown("#### 🔍 Validation Failure Details")
                    for ve in result.get("validation_errors", []):
                        st.markdown(f"**File:** `{ve.get('filename')}`")
                        st.code(ve.get("error", ""), language="text")
            else:
                st.success("🎉 Your app has been generated!")

                # ── Validation status banner ───────────────────────────────
                vs          = result.get("validation_status", "pending")
                retry_count = result.get("retry_count", 0)
                has_js      = any(
                    f.get("filename", "").endswith(".js")
                    for f in result.get("generated_files", [])
                )

                if not has_js:
                    st.info("⏭️ No JS files were generated — validation skipped.")
                elif vs == "passed" and retry_count == 0:
                    st.info("✅ JS Validation passed on first attempt.")
                elif vs == "passed" and retry_count > 0:
                    st.warning(
                        f"⚠️ JS Validation passed after **{retry_count}** fix attempt(s). "
                        "The auto-corrected files are shown below."
                    )
                # failed_exhausted is surfaced above via status=="error"

                # Build a set of failed filenames for badge display
                failed_filenames = {
                    ve.get("filename") for ve in result.get("validation_errors", [])
                }

                tabs = st.tabs(["📋 Plan", "🏗️ Architecture", "📁 Generated Files"])

                # ── Tab 1: Plan ──────────────────────────────────────────
                with tabs[0]:
                    try:
                        plan_dict = json.loads(result.get("plan", "{}"))
                        st.json(plan_dict)
                    except Exception:
                        st.code(result.get("plan", ""), language="json")

                # ── Tab 2: Architecture (Tasks) ──────────────────────────
                with tabs[1]:
                    tasks = result.get("tasks", [])
                    if tasks:
                        st.markdown(f"**{len(tasks)} files planned:**")
                        for i, task in enumerate(tasks, 1):
                            with st.expander(
                                f"📄 {task.get('filename', f'File {i}')}  —  {task.get('description', '')[:80]}",
                                expanded=(i == 1),
                            ):
                                deps = task.get("dependencies", [])
                                if deps:
                                    st.markdown(f"**Dependencies:** `{'`, `'.join(deps)}`")
                                notes = task.get("implementation_notes", "")
                                if notes:
                                    st.markdown("**Implementation notes:**")
                                    st.markdown(notes)
                    else:
                        st.info("No architecture tasks were produced.")

                # ── Tab 3: Generated Files ───────────────────────────────
                with tabs[2]:
                    files = result.get("generated_files", [])
                    if files:
                        st.markdown(
                            f'<div class="card"><b>📦 {len(files)} files generated</b></div>',
                            unsafe_allow_html=True,
                        )
                        for file_out in files:
                            fname = file_out.get("filename", "unknown")
                            lang  = file_out.get("language", "text")
                            code  = file_out.get("content", "")
                            # JS badge: ✅ if validated and passed, ⚠️ if still failing
                            is_js = fname.endswith(".js")
                            if is_js and vs == "passed":
                                badge = " ✅"
                            elif is_js and fname in failed_filenames:
                                badge = " ⚠️ (fix failed)"
                            else:
                                badge = ""
                            with st.expander(f"📄 {fname}{badge}", expanded=False):
                                st.code(code, language=lang)
                    else:
                        st.info("No files were generated.")

        except Exception as exc:
            prog_bar.progress(100, text="Pipeline stopped.")
            st.error(f"❌ Unexpected error: {exc}")
            st.exception(exc)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><small style='color:#475569'>AI App Builder · Planner → Architect → Coder · Powered by LangGraph + Groq</small></center>",
    unsafe_allow_html=True,
)
