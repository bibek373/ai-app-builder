"""
test_pipeline.py — End-to-end pipeline smoke test
Run with:  .\\venv\\Scripts\\python test_pipeline.py
"""
import json
import sys

from graph import run_pipeline

TEST_IDEA = (
    "A simple CLI to-do list app in Python with add, list, "
    "complete and delete commands, stored in a JSON file"
)

print("=" * 60)
print("END-TO-END PIPELINE TEST")
print(f"Idea: {TEST_IDEA}")
print("=" * 60)

result = run_pipeline(TEST_IDEA)

print()
print("STATUS  :", result["status"])
print("APP NAME:", result["app_name"])
print()

if result["status"] == "error":
    print("ERROR:", result["error"])
    sys.exit(1)

# ── Plan ────────────────────────────────────────────────
print("─── PLAN ───")
try:
    plan = json.loads(result["plan"])
    print(json.dumps(plan, indent=2))
except Exception:
    print(result["plan"])

# ── Tasks ───────────────────────────────────────────────
print()
print(f"─── TASKS ({len(result['tasks'])} file(s)) ───")
for t in result["tasks"]:
    print(f"  📄 {t['filename']}")
    print(f"     {t['description'][:80]}")

# ── Generated files ─────────────────────────────────────
print()
print(f"─── GENERATED FILES ({len(result['generated_files'])} file(s)) ───")
for f in result["generated_files"]:
    print(f"  ✓ {f['filename']} ({f['language']}, {len(f['content'])} chars)")

print()
print("✅ Pipeline completed successfully!")
