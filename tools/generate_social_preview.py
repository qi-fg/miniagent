"""Generate the GitHub social-preview image (1280x640) for miniagent.

Pure matplotlib, no external assets. Run:

    python tools/generate_social_preview.py

Output: assets/social-preview.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

W, H = 1280, 640

# --- background vertical gradient (#0b1220 -> #0f3460) ---------------------- #
grad = np.zeros((H, W, 3))
top = np.array([0.043, 0.071, 0.125])
bot = np.array([0.059, 0.204, 0.376])
for y in range(H):
    grad[y] = top * (1 - y / H) + bot * (y / H)

fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
ax.set_position([0, 0, 1, 1])  # full-bleed: axes fills the whole figure
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.set_aspect("auto")
ax.imshow(grad, extent=[0, W, 0, H], origin="lower", aspect="auto", zorder=0)

SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"


def box(x, y, w, h, text, fc, ec, tc="#e6f7ff", fs=15):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=12",
            fc=fc,
            ec=ec,
            lw=2,
            zorder=3,
        )
    )
    ax.text(
        x,
        y,
        text,
        fontsize=fs,
        color=tc,
        ha="center",
        va="center",
        weight="bold",
        zorder=4,
        family=SANS,
    )


def arrow(x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2),
        zorder=2,
    )


# --- title ----------------------------------------------------------------- #
ax.text(60, 556, "miniagent", fontsize=56, color="#e6f7ff", weight="bold", family=SANS)
ax.text(
    66,
    508,
    "A tiny, dependency-free LLM agent framework",
    fontsize=23,
    color="#7dd3fc",
    family=SANS,
)
ax.text(
    66,
    478,
    "ReAct loop  ·  @tool decorator  ·  transcript memory  ·  pluggable providers",
    fontsize=15,
    color="#94a3b8",
    family=SANS,
)

# --- architecture diagram --------------------------------------------------- #
box(200, 398, 175, 58, "User", "#1e293b", "#475569")
box(620, 402, 220, 70, "Agent\nReAct loop", "#0e7490", "#22d3ee")
box(1065, 398, 235, 58, "LLM Provider", "#1e293b", "#475569")
box(450, 292, 205, 58, "Memory\ntranscript", "#1e293b", "#475569")
box(800, 292, 215, 58, "Tool Registry", "#1e293b", "#475569")

arrow(290, 398, 505, 400)  # user -> agent
arrow(732, 402, 945, 398)  # agent -> provider
arrow(572, 366, 492, 320)  # agent -> memory
arrow(700, 366, 772, 320)  # agent -> tools

# --- code snippet (left) ----------------------------------------------------- #
code = (
    "from miniagent import Agent, tool, ToolRegistry, OpenAIProvider\n\n"
    "@tool\n"
    "def calculator(expr): return _safe_eval(expr)\n\n"
    "agent = Agent(OpenAIProvider(), tools=ToolRegistry([calculator]))\n"
    'print(agent.run("What is 2 ** 10 + 1?"))'
)
ax.add_patch(
    FancyBboxPatch(
        (55, 40),
        695,
        204,
        boxstyle="round,pad=0.02,rounding_size=10",
        fc="#0b1220",
        ec="#1e3a5f",
        lw=1.5,
        zorder=3,
    )
)
ax.text(
    80,
    224,
    code,
    fontsize=12,
    color="#a5f3fc",
    family=MONO,
    va="top",
    linespacing=1.45,
    zorder=4,
)

# --- feature badges (right) -------------------------------------------------- #
ax.text(795, 206, "Python 3.8+", fontsize=19, color="#e6f7ff", weight="bold", family=SANS)
ax.text(795, 172, "Zero dependencies (core)", fontsize=15, color="#7dd3fc", family=SANS)
ax.text(795, 144, "MIT License", fontsize=15, color="#7dd3fc", family=SANS)
ax.text(795, 106, "~400 lines of pure Python", fontsize=15, color="#94a3b8", family=SANS)
ax.text(795, 80, "Offline Mock demo built-in", fontsize=15, color="#94a3b8", family=SANS)
ax.text(795, 54, "OpenAI / Anthropic / Ollama", fontsize=15, color="#94a3b8", family=SANS)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "social-preview.png")
fig.savefig(out_path, dpi=100)
print(f"wrote {os.path.abspath(out_path)}")
