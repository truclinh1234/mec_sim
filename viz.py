# =============================================================================
# viz.py — Vẽ quá trình của task từ CSV metrics
# Chạy sau main.py: python viz.py
# Xuất: results/task_breakdown.png  (quá trình từng task)
#        results/queue_timeline.png  (queue theo thời gian)
#        results/latency_dist.png    (phân phối latency)
# =============================================================================
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config as cfg

OUTDIR   = cfg.METRICS_OUTPUT_DIR
RUN_NAME = "sim"


def load_tasks():
    path = os.path.join(OUTDIR, f"{RUN_NAME}_tasks.csv")
    return list(csv.DictReader(open(path)))

def load_timeseries():
    path = os.path.join(OUTDIR, f"{RUN_NAME}_timeseries.csv")
    return list(csv.DictReader(open(path)))

def load_summary():
    path = os.path.join(OUTDIR, f"{RUN_NAME}_summary.json")
    return json.load(open(path))


# ─── [1] Gantt: quá trình từng task (tx → wait → proc) ─────────────────────
def plot_task_breakdown(rows, max_tasks=40):
    """
    Mỗi task 1 thanh ngang, chia 3 đoạn màu:
      xanh dương = tx_delay (truyền lên edge, = 0 nếu local)
      vàng       = waiting  (chờ trong queue CPU)
      xanh lá   = proc      (CPU xử lý)
    Trục x = thời gian (ms), trục y = task_id
    """
    done = [r for r in rows if r["latency_ms"]]
    sample = done[:max_tasks]

    fig, ax = plt.subplots(figsize=(13, max(4, len(sample) * 0.28)))

    for i, r in enumerate(sample):
        at   = float(r["arrival_time"]) * 1000   # ms
        tx   = float(r["tx_delay_ms"])
        wait = float(r["waiting_ms"])  if r["waiting_ms"]  else 0
        proc = float(r["proc_ms"])     if r["proc_ms"]     else 0

        y    = i
        h    = 0.6
        # tx
        if tx > 0:
            ax.barh(y, tx,   left=at,        height=h, color="#3b82f6", alpha=0.85)
        # wait
        ax.barh(y, wait, left=at + tx,    height=h, color="#f59e0b", alpha=0.85)
        # proc
        ax.barh(y, proc, left=at+tx+wait, height=h, color="#22c55e", alpha=0.85)

        # nhãn task type + dest
        dest = f"→E{r['edge_id']}" if r["offloaded"] == "1" else "local"
        label = f"U{r['user_id']} {r['task_type']} {dest}"
        ax.text(at - 1, y, label, va="center", ha="right", fontsize=6.5,
                color="#374151")

    ax.set_xlabel("Simulation time (ms)")
    ax.set_title(f"Task breakdown — {len(sample)} tasks\n"
                 "■ tx_delay  ■ waiting  ■ proc",
                 fontsize=10)
    ax.set_yticks([])
    ax.invert_yaxis()

    patches = [
        mpatches.Patch(color="#3b82f6", label="tx_delay (truyền lên edge)"),
        mpatches.Patch(color="#f59e0b", label="waiting (chờ queue CPU)"),
        mpatches.Patch(color="#22c55e", label="proc (CPU xử lý)"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    out = os.path.join(OUTDIR, "task_breakdown.png")
    fig.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [viz] task_breakdown → {out}")


# ─── [2] Queue length theo thời gian ────────────────────────────────────────
def plot_queue_timeline(ts_rows):
    t     = [float(r["sim_time"])        for r in ts_rows]
    u_q   = [float(r["total_user_qlen"]) for r in ts_rows]
    e_q   = [float(r["total_edge_qlen"]) for r in ts_rows]
    lat   = [float(r["mean_lat_all_ms"]) if r["mean_lat_all_ms"] else 0
             for r in ts_rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax1.fill_between(t, u_q, alpha=0.4, color="#3b82f6", label="User queues")
    ax1.fill_between(t, e_q, alpha=0.4, color="#f59e0b", label="Edge queues")
    ax1.plot(t, u_q, color="#3b82f6", lw=1.5)
    ax1.plot(t, e_q, color="#f59e0b", lw=1.5)
    ax1.set_ylabel("Tasks in queue")
    ax1.set_title("Queue length over time")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(t, lat, color="#8b5cf6", lw=1.5)
    ax2.fill_between(t, lat, alpha=0.2, color="#8b5cf6")
    ax2.set_ylabel("Mean latency (ms)")
    ax2.set_xlabel("Simulation time (s)")
    ax2.set_title("Mean latency over time")
    ax2.grid(alpha=0.3)

    out = os.path.join(OUTDIR, "queue_timeline.png")
    fig.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [viz] queue_timeline  → {out}")


# ─── [3] Phân phối latency theo loại task và routing ───────────────────────
def plot_latency_dist(rows):
    done = [r for r in rows if r["latency_ms"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Subplot trái: stacked bar trung bình tx / wait / proc ──────────────
    ax = axes[0]
    labels, txs, waits, procs = [], [], [], []

    groups = [
        ("Light  local",  "Light",  "0"),
        ("Light  edge",   "Light",  "1"),
        ("Medium local",  "Medium", "0"),
        ("Medium edge",   "Medium", "1"),
        ("Heavy  edge",   "Heavy",  "1"),
    ]
    for lbl, ttype, off in groups:
        sub = [r for r in done
               if r["task_type"] == ttype and r["offloaded"] == off]
        if not sub:
            continue
        labels.append(lbl)
        txs.append(  np.mean([float(r["tx_delay_ms"]) for r in sub]))
        waits.append( np.mean([float(r["waiting_ms"])  if r["waiting_ms"]  else 0 for r in sub]))
        procs.append( np.mean([float(r["proc_ms"])     if r["proc_ms"]     else 0 for r in sub]))

    x = np.arange(len(labels))
    ax.bar(x, txs,   label="tx_delay",  color="#3b82f6", alpha=0.85)
    ax.bar(x, waits, label="waiting",   color="#f59e0b", alpha=0.85, bottom=txs)
    ax.bar(x, procs, label="proc",      color="#22c55e", alpha=0.85,
           bottom=[t+w for t,w in zip(txs, waits)])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("ms"); ax.set_title("Mean latency breakdown\nby type & routing")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # ── Subplot phải: histogram latency local vs edge ──────────────────────
    ax = axes[1]
    local_lats = [float(r["latency_ms"]) for r in done if r["offloaded"] == "0"]
    edge_lats  = [float(r["latency_ms"]) for r in done if r["offloaded"] == "1"]
    bins = np.linspace(0, max(local_lats + edge_lats + [1]), 30)
    ax.hist(local_lats, bins=bins, alpha=0.6, color="#22c55e", label=f"Local (n={len(local_lats)})")
    ax.hist(edge_lats,  bins=bins, alpha=0.6, color="#f59e0b", label=f"Edge  (n={len(edge_lats)})")
    ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count")
    ax.set_title("Latency distribution\nlocal vs edge")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    out = os.path.join(OUTDIR, "latency_dist.png")
    fig.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [viz] latency_dist    → {out}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rows = load_tasks()
    ts   = load_timeseries()
    print(f"Loaded {len(rows)} tasks, {len(ts)} timeseries rows")
    plot_task_breakdown(rows)
    plot_queue_timeline(ts)
    plot_latency_dist(rows)
    print("Done.")
