import os
import subprocess

os.chdir(r"E:\AGENT\WorkBuddy\miniagent")
env = dict(os.environ)
env["https_proxy"] = "http://127.0.0.1:7890"
env["http_proxy"] = "http://127.0.0.1:7890"
GH = r"E:\AGENT\WorkBuddy\bin\gh\bin\gh.exe"

# cleanup temp files before staging
for f in ("_runner.py", "_smoke.log"):
    try:
        os.remove(f)
    except OSError:
        pass


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


steps = [
    ("gh auth setup-git", run([GH, "auth", "setup-git"])),
    ("git add -A", run(["git", "add", "-A"])),
    ("git commit", run(["git", "commit", "-m",
        "feat: add real_agent.py chat loop + social preview image\n\n"
        "- examples/real_agent.py: interactive chat loop for OpenAI/Anthropic/Ollama/Mock\n"
        "  with calculator (ast-safe), current_time, count_words tools; sys.path shim\n"
        "  so it runs directly from a fresh clone\n"
        "- assets/social-preview.png + tools/generate_social_preview.py: 1280x640 cover\n"
        "- README: cover image, 'Talk to a real model' section, updated project layout"])),
    ("git push", run(["git", "push", "origin", "main"])),
    ("git log -1", run(["git", "log", "--oneline", "-1"])),
    ("git status", run(["git", "status", "--short"])),
]

with open("_push.log", "w", encoding="utf-8") as f:
    for name, r in steps:
        f.write(f"===== {name} (exit {r.returncode}) =====\n")
        f.write((r.stdout or "") + (r.stderr or "") + "\n")
print("done")
