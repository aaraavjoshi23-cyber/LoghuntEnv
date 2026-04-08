# -*- coding: utf-8 -*-
"""
LogHuntEnv -- Pre-Submission Validator
Run this before submitting: python validate.py
"""
import sys
import os
import subprocess
import time

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PASS_MARK = "[PASS]"
FAIL_MARK = "[FAIL]"
results = []

def check(name, ok, detail=""):
    status = PASS_MARK if ok else FAIL_MARK
    results.append((name, ok, detail))
    msg = f"  {status}  {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)

# ── 1. Check required files exist ────────────────────────────────────────────
print("\n[1] Checking required files...")
required_files = [
    "inference.py", "app.py", "env.py", "openenv.yaml",
    "Dockerfile", "readme.md", "requirements.txt", "create_dataset.py"
]
for f in required_files:
    check(f"File exists: {f}", os.path.exists(f))

# ── 2. Validate openenv.yaml ──────────────────────────────────────────────────
print("\n[2] Validating openenv.yaml...")
try:
    import yaml
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    check("Has 'name' field", "name" in spec)
    check("Has 'version' field", "version" in spec)
    check("Has 'observation_space'", "observation_space" in spec)
    check("Has 'action_space'", "action_space" in spec)
    check("Has 'tasks'", "tasks" in spec)
    check("Has 3+ tasks", len(spec.get("tasks", [])) >= 3)
    check("Has 'reward' field", "reward" in spec)
    tasks = spec.get("tasks", [])
    task_ids = [t["id"] for t in tasks]
    check("Has 'easy' task", "easy"   in task_ids)
    check("Has 'medium' task", "medium" in task_ids)
    check("Has 'hard' task", "hard"   in task_ids)
    for t in tasks:
        check(f"Task '{t['id']}' has reward_threshold", "reward_threshold" in t)
except Exception as e:
    check("openenv.yaml is valid YAML", False, str(e))

# ── 3. Check .env variables ───────────────────────────────────────────────────
print("\n[3] Checking environment variables...")
from dotenv import load_dotenv
load_dotenv()
check("API_BASE_URL is set", bool(os.environ.get("API_BASE_URL")))
check("MODEL_NAME is set",   bool(os.environ.get("MODEL_NAME")))
check("HF_TOKEN is set",     bool(os.environ.get("HF_TOKEN")))

# ── 4. Check Python imports ───────────────────────────────────────────────────
print("\n[4] Checking Python package imports...")
packages = {
    "numpy":        "numpy",
    "pandas":       "pandas",
    "gymnasium":    "gymnasium",
    "fastapi":      "fastapi",
    "openai":       "openai",
    "python-dotenv":"dotenv",
    "pyyaml":       "yaml",
    "requests":     "requests",
}
for pkg_name, import_name in packages.items():
    try:
        __import__(import_name)
        check(f"{pkg_name} importable", True)
    except ImportError as e:
        check(f"{pkg_name} importable", False, str(e))

# ── 5. Generate dataset if missing ────────────────────────────────────────────
if not os.path.exists("data/CICIDS2017_sample.csv"):
    print("\n  Generating dataset...")
    subprocess.run([sys.executable, "create_dataset.py"], capture_output=True)

# ── 6. Spin up FastAPI server and test endpoints ──────────────────────────────
print("\n[5] Starting FastAPI server for endpoint tests (wait 5s)...")
import requests as req

server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app:app",
     "--host", "127.0.0.1", "--port", "8765"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
time.sleep(5)

BASE = "http://127.0.0.1:8765"
try:
    # Health check
    r = req.get(f"{BASE}/", timeout=5)
    check("GET /  -> 200", r.status_code == 200, f"status={r.status_code}")
    check("GET / returns status=ok", r.json().get("status") == "ok")

    r = req.get(f"{BASE}/health", timeout=5)
    check("GET /health -> 200", r.status_code == 200)

    r = req.get(f"{BASE}/tasks", timeout=5)
    check("GET /tasks -> 200", r.status_code == 200)
    check("GET /tasks returns 3 tasks", len(r.json()) == 3)

    # Reset for each task
    for tid in ["easy", "medium", "hard"]:
        r = req.post(f"{BASE}/reset", json={"task_id": tid}, timeout=10)
        check(f"POST /reset task_id='{tid}' -> 200", r.status_code == 200)
        obs = r.json().get("observation", [])
        check(f"  Observation length = 202 for '{tid}'", len(obs) == 202, f"got {len(obs)}")

    # Reset to easy for step tests
    req.post(f"{BASE}/reset", json={"task_id": "easy"}, timeout=10)

    # Step tests -- all 6 actions
    for action_id in range(6):
        r = req.post(f"{BASE}/step", json={"action": action_id}, timeout=10)
        check(f"POST /step action={action_id} -> 200", r.status_code == 200)
        if r.status_code == 200:
            data = r.json()
            check(f"  Step {action_id} has 'reward'", "reward" in data)
            check(f"  Step {action_id} has 'done'",   "done"   in data)

    r = req.get(f"{BASE}/state", timeout=5)
    check("GET /state -> 200", r.status_code == 200)
    check("GET /state has 'observation'", "observation" in r.json())

except Exception as e:
    check("API server tests", False, str(e))
finally:
    server_proc.terminate()
    server_proc.wait()

# ── 7. Check inference.py log format ─────────────────────────────────────────
print("\n[6] Checking inference.py format...")
try:
    with open("inference.py") as f:
        code = f.read()
    check("Has [START] log type",     '"[START]"' in code)
    check("Has [STEP] log type",      '"[STEP]"'  in code)
    check("Has [END] log type",       '"[END]"'   in code)
    check("Uses OpenAI client",       "OpenAI("   in code)
    check("Uses API_BASE_URL",        "API_BASE_URL" in code)
    check("Uses MODEL_NAME",          "MODEL_NAME"   in code)
    check("Uses HF_TOKEN",            "HF_TOKEN"     in code)
    check("Handles 3 tasks",          '["easy", "medium", "hard"]' in code)
except Exception as e:
    check("inference.py readable", False, str(e))

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
passed = sum(1 for _, ok, _ in results if ok)
total  = len(results)
pct    = int(100 * passed / total) if total else 0
print(f"  Result: {passed}/{total} checks passed ({pct}%)")
if passed == total:
    print("  *** ALL CHECKS PASSED -- Ready to submit! ***")
else:
    failed = [name for name, ok, _ in results if not ok]
    print("  Fix these before submitting:")
    for name in failed:
        print(f"    - {name}")
print("=" * 55 + "\n")
sys.exit(0 if passed == total else 1)
