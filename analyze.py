"""
Analysis & Visualisation
========================
Runs both methods (with optuna best params if available), then produces:
  1. Reward curves (smoothed)
  2. Violation rate curves
  3. Lambda schedule (Lagrangian)
  4. Pareto frontier: reward vs. violation rate
  5. Best-episode grid replay saved as PNG strip
  6. Best-episode step log (console + CSV)
  7. Summary comparison table
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import pandas as pd
import json, os, cv2, time

import method1_lagrangian   as m1
import method2_safety_layer as m2
from config import *

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMOOTH_WIN = 40       # rolling-average window for plots


# ─── helpers ────────────────────────────────────────────────────────────────

def smooth(arr, w=SMOOTH_WIN):
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


def load_best_params():
    path = "best_params.json"
    if not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        data = json.load(f)
    return (data.get("lagrangian",   {}).get("best_params", {}),
            data.get("safety_layer", {}).get("best_params", {}))


# ─── 1. train both methods ───────────────────────────────────────────────────

def run_training():
    lag_kw, sl_kw = load_best_params()
    if lag_kw:
        print(f"Using Optuna params for Lagrangian:    {lag_kw}")
    if sl_kw:
        print(f"Using Optuna params for Safety-Layer:  {sl_kw}")

    print("\n── Training Method 1: Lagrangian RL ──")
    t0 = time.time()
    h1, agent1 = m1.train(agent_kwargs=lag_kw or None)
    print(f"   Done in {time.time()-t0:.1f}s")

    print("\n── Training Method 2: Safety Layer ──")
    t0 = time.time()
    h2, agent2 = m2.train(agent_kwargs=sl_kw or None)
    print(f"   Done in {time.time()-t0:.1f}s")

    return h1, h2


# ─── 2. reward + violation curves ───────────────────────────────────────────

def plot_learning_curves(h1, h2):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Safe RL — Learning Curves", fontsize=15, fontweight="bold")

    eps = np.array(h1["episode"])

    # reward
    for ax, (h, lbl, c) in zip(
            axes[0],
            [(h1, "Lagrangian RL", "#E63946"),
             (h2, "Safety Layer",  "#457B9D")]):
        r = np.array(h["reward"])
        ax.plot(eps, r, alpha=0.2, color=c)
        ax.plot(eps, smooth(r), color=c, lw=2, label=lbl)
        ax.set_title(f"{lbl} — Reward")
        ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
        ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.legend(); ax.grid(True, alpha=0.3)

    # violation rate
    for ax, (h, lbl, c) in zip(
            axes[1],
            [(h1, "Lagrangian RL", "#E63946"),
             (h2, "Safety Layer",  "#457B9D")]):
        v = np.array(h["violation_rate"])
        ax.plot(eps, v, alpha=0.2, color=c)
        ax.plot(eps, smooth(v), color=c, lw=2, label=lbl)
        ax.set_title(f"{lbl} — Violation Rate")
        ax.set_xlabel("Episode"); ax.set_ylabel("Violation Rate")
        ax.axhline(LAG_COST_LIMIT, ls="--", color="black",
                   lw=1.2, label="cost limit")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "learning_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ─── 3. lambda schedule ──────────────────────────────────────────────────────

def plot_lambda(h1):
    fig, ax = plt.subplots(figsize=(9, 4))
    lam = np.array(h1["lambda"])
    eps = np.array(h1["episode"])
    ax.plot(eps, lam, color="#E63946", lw=1.5, label="λ (dual variable)")
    ax.fill_between(eps, 0, lam, alpha=0.15, color="#E63946")
    ax.set_title("Lagrangian RL — Dual Variable λ Over Training",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode"); ax.set_ylabel("λ")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lambda_schedule.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ─── 4. Pareto analysis ──────────────────────────────────────────────────────

def pareto_analysis(h1, h2):
    """
    For each window of SMOOTH_WIN episodes compute (avg_reward, avg_viol_rate).
    Plot as scatter — Pareto-optimal points highlighted.
    """
    def window_points(h, w=SMOOTH_WIN):
        r = np.array(h["reward"])
        v = np.array(h["violation_rate"])
        pts = []
        for i in range(0, len(r) - w, w // 2):
            pts.append((np.mean(r[i:i+w]), np.mean(v[i:i+w])))
        return np.array(pts)

    def pareto_front(pts):
        """Return mask of Pareto-optimal points (max reward, min violation)."""
        mask = np.ones(len(pts), dtype=bool)
        for i, (r_i, v_i) in enumerate(pts):
            for j, (r_j, v_j) in enumerate(pts):
                if i == j:
                    continue
                if r_j >= r_i and v_j <= v_i:
                    mask[i] = False
                    break
        return mask

    pts1 = window_points(h1)
    pts2 = window_points(h2)

    fig, ax = plt.subplots(figsize=(9, 6))

    for pts, lbl, c, mk in [(pts1, "Lagrangian RL", "#E63946", "o"),
                              (pts2, "Safety Layer",  "#457B9D", "s")]:
        pf = pareto_front(pts)
        ax.scatter(pts[:,1], pts[:,0], color=c, alpha=0.35,
                   marker=mk, s=50, label=f"{lbl} (all windows)")
        ax.scatter(pts[pf,1], pts[pf,0], color=c, edgecolors="black",
                   linewidths=1.5, s=120, marker=mk,
                   label=f"{lbl} Pareto-optimal")
        # connect Pareto front
        sorted_pf = pts[pf][np.argsort(pts[pf, 1])]
        ax.plot(sorted_pf[:,1], sorted_pf[:,0], color=c,
                lw=1.5, ls="--", alpha=0.7)

    ax.axvline(LAG_COST_LIMIT, ls=":", color="gray", lw=1.2,
               label=f"cost limit ({LAG_COST_LIMIT})")
    ax.set_xlabel("Average Violation Rate  ↓  (safer →)", fontsize=11)
    ax.set_ylabel("Average Reward  ↑  (better →)", fontsize=11)
    ax.set_title("Pareto Analysis: Reward vs Safety", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pareto.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ─── 5. best-episode grid replay ─────────────────────────────────────────────

CELL = 40    # pixels per grid cell for replay render

def _colour_map():
    """Return fixed RGB colours matching env.render logic."""
    return {
        "bg":      (255, 218, 185),
        "wall":    (120, 120, 120),
        "hazard":  (176, 224, 230),
        "human":   (255, 140,   0),
        "goal":    (  0, 255,   0),
        "agent":   ( 50, 205,  50),
    }


def save_best_episode_strip(history, method_name, max_frames=12):
    frames = history.get("best_frames")
    if not frames:
        print(f"No frames for {method_name}")
        return

    # pick evenly spaced frames
    idxs = np.linspace(0, len(frames)-1, min(max_frames, len(frames)),
                        dtype=int)
    selected = [frames[i] for i in idxs]

    size = GRID_SIZE * CELL
    strip_frames = []
    for f in selected:
        img = cv2.resize(f, (size, size), interpolation=cv2.INTER_NEAREST)
        strip_frames.append(img)

    # horizontal strip
    strip = np.concatenate(strip_frames, axis=1)

    # add title bar
    bar = np.ones((40, strip.shape[1], 3), dtype=np.uint8) * 30
    label = f"Best Episode — {method_name}   (reward={history['best_reward']:.1f})"
    cv2.putText(bar, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    strip = np.vstack([bar, strip])

    path = os.path.join(OUTPUT_DIR,
                        f"best_episode_{method_name.replace(' ','_')}.png")
    cv2.imwrite(path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"Saved → {path}")


# ─── 6. best-episode step log ────────────────────────────────────────────────

def save_best_episode_log(history, method_name):
    log = history.get("best_log", [])
    if not log:
        return

    action_names = ["UP","DOWN","LEFT","RIGHT","STAY"]
    rows = []
    for entry in log:
        row = {
            "step":       entry["step"],
            "pos":        str(entry["pos"]),
            "reward":     round(entry["reward"], 2),
            "cost":       entry["cost"],
            "hazard":     entry["info"].get("hazard", False),
            "collision":  entry["info"].get("collision", False),
            "proximity":  entry["info"].get("distance", False),
        }
        if "action" in entry:
            row["action"] = action_names[entry["action"]]
        else:
            row["proposed"]  = action_names[entry["proposed"]]
            row["safe_action"]= action_names[entry["safe_action"]]
            row["intervened"] = entry["intervened"]
        if "lambda" in entry:
            row["lambda"] = round(entry["lambda"], 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    fname = f"best_episode_log_{method_name.replace(' ','_')}.csv"
    path  = os.path.join(OUTPUT_DIR, fname)
    df.to_csv(path, index=False)
    print(f"Saved → {path}")
    print(f"\n{'─'*60}")
    print(f"  Best episode log — {method_name}")
    print(f"  Episode {history['best_episode']}  |  "
          f"reward = {history['best_reward']:.2f}")
    print(f"{'─'*60}")
    print(df.to_string(index=False))


# ─── 7. combined reward + violation bar chart ────────────────────────────────

def plot_comparison_bar(h1, h2):
    last = 200
    metrics = {
        "Avg Reward\n(last 200 ep)":         [np.mean(h1["reward"][-last:]),
                                               np.mean(h2["reward"][-last:])],
        "Avg Violation Rate\n(last 200 ep)": [np.mean(h1["violation_rate"][-last:]),
                                               np.mean(h2["violation_rate"][-last:])],
        "Success Rate\n(last 200 ep)":       [np.mean(h1["success"][-last:]),
                                               np.mean(h2["success"][-last:])],
        "Avg Steps\n(last 200 ep)":          [np.mean(h1["steps"][-last:]),
                                               np.mean(h2["steps"][-last:])],
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Method Comparison (last 200 episodes)", fontsize=13,
                 fontweight="bold")
    colors = ["#E63946", "#457B9D"]
    methods = ["Lagrangian RL", "Safety Layer"]

    for ax, (metric, vals) in zip(axes, metrics.items()):
        bars = ax.bar(methods, vals, color=colors, edgecolor="white",
                      linewidth=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + abs(bar.get_height())*0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
        ax.set_title(metric, fontsize=10)
        ax.tick_params(axis="x", labelrotation=10)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "comparison_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ─── 8. summary table ────────────────────────────────────────────────────────

def print_summary(h1, h2):
    last = 200
    rows = []
    for h, name in [(h1, "Lagrangian RL"), (h2, "Safety Layer")]:
        rows.append({
            "Method":           name,
            "Avg Reward":       f"{np.mean(h['reward'][-last:]):.2f}",
            "Avg Viol Rate":    f"{np.mean(h['violation_rate'][-last:]):.4f}",
            "Success Rate":     f"{np.mean(h['success'][-last:]):.3f}",
            "Avg Steps":        f"{np.mean(h['steps'][-last:]):.1f}",
            "Best Reward":      f"{h['best_reward']:.2f}",
            "Best Episode":     h['best_episode'],
        })
    df = pd.DataFrame(rows).set_index("Method")
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(df.to_string())
    print("="*70)

    path = os.path.join(OUTPUT_DIR, "summary.csv")
    df.to_csv(path)
    print(f"Saved → {path}")


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    h1, h2 = run_training()

    plot_learning_curves(h1, h2)
    plot_lambda(h1)
    pareto_analysis(h1, h2)
    plot_comparison_bar(h1, h2)

    save_best_episode_strip(h1, "Lagrangian_RL")
    save_best_episode_strip(h2, "Safety_Layer")

    save_best_episode_log(h1, "Lagrangian_RL")
    save_best_episode_log(h2, "Safety_Layer")

    print_summary(h1, h2)
    print(f"\nAll outputs saved to  ./{OUTPUT_DIR}/")
