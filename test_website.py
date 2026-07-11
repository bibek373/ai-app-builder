"""
test_website.py — Static website generation test
Verifies:
  1. No placeholder text like [Insert ...] remains in generated HTML
  2. Design-quality CSS patterns are present (section separation, reveal animations, etc.)
"""
import json
import re
import sys

from graph import run_pipeline

TEST_IDEA = (
    "A single-page website about the benefits of artificial intelligence "
    "in everyday life. Include sections for: hero header, key benefits, "
    "real-world applications, and a call-to-action footer."
)

print("=" * 60)
print("STATIC WEBSITE GENERATION TEST")
print(f"Idea: {TEST_IDEA[:80]}...")
print("=" * 60)

result = run_pipeline(TEST_IDEA)

print()
print("STATUS  :", result["status"])
print("APP NAME:", result["app_name"])
print()

if result["status"] == "error":
    print("ERROR:", result["error"])
    sys.exit(1)

# ── Check generated files ───────────────────────────────
files = result.get("generated_files", [])
print(f"─── GENERATED FILES ({len(files)}) ───")
for f in files:
    print(f"  ✓ {f['filename']} ({f['language']}, {len(f['content'])} chars)")

# ── Quality checks on HTML files ────────────────────────
PLACEHOLDER_RE = re.compile(
    r'\['
    r'(?:Insert|Add|Your|Placeholder|Enter|Replace|Put|Write|Fill|Update|Include|Provide)'
    r'\s[^\]]{3,}'
    r'\]',
    re.IGNORECASE,
)

html_files = [f for f in files if f["filename"].endswith((".html", ".htm"))]
css_files  = [f for f in files if f["filename"].endswith(".css")]

print()
print("─── QUALITY CHECKS ───")

all_ok = True

for hf in html_files:
    placeholders = PLACEHOLDER_RE.findall(hf["content"])
    if placeholders:
        print(f"  ❌ {hf['filename']}: Found {len(placeholders)} placeholder(s):")
        for p in placeholders:
            print(f"      → {p}")
        all_ok = False
    else:
        print(f"  ✅ {hf['filename']}: No placeholder text found")

    # Check for design quality markers
    content_lower = hf["content"].lower()
    checks = {
        "max-width container": "max-width" in content_lower,
        "reveal class":        "reveal" in content_lower,
    }
    for label, ok in checks.items():
        sym = "✅" if ok else "⚠️"
        print(f"  {sym} {hf['filename']}: {label} {'present' if ok else 'MISSING'}")
        if not ok:
            all_ok = False

for cf in css_files:
    content_lower = cf["content"].lower()
    checks = {
        "Google Fonts import":  "@import" in content_lower or "fonts.googleapis" in content_lower,
        ".reveal class":        ".reveal" in content_lower,
        "box-shadow":           "box-shadow" in content_lower,
        "border-radius":        "border-radius" in content_lower,
        "transition":           "transition" in content_lower,
    }
    for label, ok in checks.items():
        sym = "✅" if ok else "⚠️"
        print(f"  {sym} {cf['filename']}: {label} {'present' if ok else 'MISSING'}")
        if not ok:
            all_ok = False

js_files = [f for f in files if f["filename"].endswith(".js")]
for jf in js_files:
    has_observer = "intersectionobserver" in jf["content"].lower()
    sym = "✅" if has_observer else "⚠️"
    print(f"  {sym} {jf['filename']}: IntersectionObserver {'present' if has_observer else 'MISSING'}")
    if not has_observer:
        all_ok = False

print()
if all_ok:
    print("✅ All quality checks passed!")
else:
    print("⚠️  Some quality checks failed — see above.")
