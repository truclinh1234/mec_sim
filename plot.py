# =============================================================================
# plot.py — Vẽ toàn bộ đồ thị báo cáo từ results/
#
# Yêu cầu:  pip install matplotlib pandas seaborn numpy
# Sử dụng:  python plot.py
# =============================================================================
import os
import json
import math
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.dpi":       100,
    "savefig.dpi":      300,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

RESULTS_DIR = "results"
RUN_NAME    = "sim"

# ── Màu sắc nhất quán ─────────────────────────────────────────────────────────
C_UCB     = "#4361EE"   # xanh đậm  – UCB
C_LOCAL   = "#F77F00"   # cam       – Local
C_EDGE0   = "#3A86FF"   # xanh nhạt – Edge 0
C_EDGE1   = "#06D6A0"   # xanh lá   – Edge 1
C_REWARD  = "#7209B7"   # tím       – reward

APP_COLORS = {
    "lightgbm":  "#4361EE",
    "mapreduce": "#F77F00",
    "matrix":    "#06D6A0",
    "video":     "#EF233C",
}
APP_LABELS = {
    "lightgbm":  "LightGBM",
    "mapreduce": "MapReduce",
    "matrix":    "Matrix App",
    "video":     "Video App",
}

# =============================================================================
# Helpers
# =============================================================================

def load_data():
    ts_path   = os.path.join(RESULTS_DIR, f"{RUN_NAME}_timeseries.csv")
    task_path = os.path.join(RESULTS_DIR, f"{RUN_NAME}_tasks.csv")
    sum_path  = os.path.join(RESULTS_DIR, f"{RUN_NAME}_summary.json")

    if not os.path.exists(ts_path):
        raise FileNotFoundError(
            f"Không tìm thấy {ts_path}. Hãy chạy main.py trước."
        )

    ts   = pd.read_csv(ts_path)
    task = pd.read_csv(task_path) if os.path.exists(task_path) else pd.DataFrame()
    with open(sum_path) as f:
        summary = json.load(f)

    print(f"[plot] Đã nạp: {len(ts)} snapshots, {len(task)} tasks")
    return ts, task, summary


def savefig(fig, name: str):
    path = os.path.join(RESULTS_DIR, f"fig_{name}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  → {path}")
    plt.close(fig)


def rolling_mean(series: pd.Series, w: int = 5) -> pd.Series:
    return series.rolling(w, min_periods=1).mean()


# =============================================================================
# Đồ thị 1: UCB Reward theo thời gian + Cumulative Reward
# =============================================================================

def plot_reward_convergence(ts: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- 1a. Reward trung bình mỗi interval ---
    ax = axes[0]
    reward_col = "ucb_reward_mean"
    if reward_col in ts.columns:
        valid = ts.dropna(subset=[reward_col])
        ax.plot(valid["sim_time"], valid[reward_col],
                color=C_UCB, alpha=0.35, linewidth=1, label="Mean reward (raw)")
        ax.plot(valid["sim_time"], rolling_mean(valid[reward_col]),
                color=C_UCB, linewidth=2, label="Rolling mean (w=5)")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Thời gian mô phỏng (s)")
        ax.set_ylabel("Reward trung bình")
        ax.set_title("(a) UCB Reward theo thời gian")
        ax.legend()
    else:
        ax.set_title("(a) Không có cột ucb_reward_mean")

    # --- 1b. Cumulative reward (convergence) ---
    ax = axes[1]
    cum_col = "ucb_reward_cumulative"
    if cum_col in ts.columns:
        valid = ts.dropna(subset=[cum_col])
        ax.plot(valid["sim_time"], valid[cum_col],
                color=C_REWARD, linewidth=2)
        ax.fill_between(valid["sim_time"], 0, valid[cum_col],
                        alpha=0.12, color=C_REWARD)
        ax.set_xlabel("Thời gian mô phỏng (s)")
        ax.set_ylabel("Cumulative reward")
        ax.set_title("(b) Cumulative Reward (hội tụ UCB)")
    else:
        ax.set_title("(b) Không có cột ucb_reward_cumulative")

    fig.suptitle("Đồ thị 1 — Học tập & Hội tụ UCB", fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "1_reward_convergence")


# =============================================================================
# Đồ thị 2: Latency theo thời gian (all / local / edge)
# =============================================================================

def plot_latency_timeseries(ts: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4.5))

    cols = {
        "mean_lat_all_ms":   ("Tất cả",  C_UCB,   2.0),
        "mean_lat_local_ms": ("Local",   C_LOCAL, 1.4),
        "mean_lat_edge_ms":  ("Edge",    C_EDGE0, 1.4),
    }
    for col, (label, color, lw) in cols.items():
        if col in ts.columns:
            valid = ts.dropna(subset=[col])
            ax.plot(valid["sim_time"], rolling_mean(valid[col]),
                    color=color, linewidth=lw, label=label)

    if "p95_lat_all_ms" in ts.columns:
        valid = ts.dropna(subset=["p95_lat_all_ms"])
        ax.fill_between(valid["sim_time"],
                        rolling_mean(valid.get("mean_lat_all_ms", 0)),
                        rolling_mean(valid["p95_lat_all_ms"]),
                        alpha=0.10, color=C_UCB, label="p95 band")

    ax.set_xlabel("Thời gian mô phỏng (s)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Đồ thị 2 — Latency theo thời gian (rolling mean)")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "2_latency_timeseries")


# =============================================================================
# Đồ thị 3: So sánh latency theo loại app (bar chart)
# =============================================================================

def plot_latency_by_app(summary: dict):
    by_type = summary.get("by_type", {})
    if not by_type:
        print("[plot] Bỏ qua đồ thị 3: không có by_type trong summary")
        return

    apps  = list(by_type.keys())
    means = [by_type[a]["latency_ms"].get("mean", 0) for a in apps]
    p95s  = [by_type[a]["latency_ms"].get("p95",  0) for a in apps]
    colors = [APP_COLORS.get(a, "#888") for a in apps]
    labels = [APP_LABELS.get(a, a)      for a in apps]

    x  = np.arange(len(apps))
    w  = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))

    bars_mean = ax.bar(x - w/2, means, w, label="Mean latency",
                       color=colors, alpha=0.85, edgecolor="white")
    bars_p95  = ax.bar(x + w/2, p95s,  w, label="p95 latency",
                       color=colors, alpha=0.45, edgecolor="white", hatch="//")

    # Nhãn giá trị trên cột
    for bar in bars_mean:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                f"{h:.0f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_p95:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2,
                f"{h:.0f}", ha="center", va="bottom", fontsize=9, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Đồ thị 3 — Latency trung bình & p95 theo ứng dụng DAG")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "3_latency_by_app")


# =============================================================================
# Đồ thị 4: Offloading decision — phân phối arm (pie + bar)
# =============================================================================

def plot_arm_distribution(task_df: pd.DataFrame):
    if task_df.empty or "ucb_arm" not in task_df.columns:
        print("[plot] Bỏ qua đồ thị 4: không có cột ucb_arm")
        return

    arm_counts = task_df["ucb_arm"].dropna().value_counts()
    if arm_counts.empty:
        print("[plot] Bỏ qua đồ thị 4: ucb_arm rỗng")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- 4a. Pie chart ---
    arm_colors = []
    for arm in arm_counts.index:
        if arm == "Local":
            arm_colors.append(C_LOCAL)
        elif "0" in str(arm):
            arm_colors.append(C_EDGE0)
        else:
            arm_colors.append(C_EDGE1)

    axes[0].pie(arm_counts.values, labels=arm_counts.index,
                colors=arm_colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white"})
    axes[0].set_title("(a) Tỷ lệ quyết định offloading")

    # --- 4b. Theo app ---
    if "app_name" in task_df.columns:
        pivot = (task_df.dropna(subset=["ucb_arm"])
                 .groupby(["app_name", "ucb_arm"])
                 .size().unstack(fill_value=0))
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
        pivot_pct.index = [APP_LABELS.get(i, i) for i in pivot_pct.index]
        pivot_pct.plot(kind="bar", ax=axes[1], colormap="tab10",
                       edgecolor="white", alpha=0.85)
        axes[1].set_ylabel("Tỷ lệ (%)")
        axes[1].set_xlabel("")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha="right")
        axes[1].set_title("(b) Quyết định offloading theo ứng dụng")
        axes[1].legend(title="Arm", loc="upper right")
    else:
        axes[1].set_visible(False)

    fig.suptitle("Đồ thị 4 — Phân phối quyết định UCB (Arm Selection)", fontweight="bold")
    fig.tight_layout()
    savefig(fig, "4_arm_distribution")


# =============================================================================
# Đồ thị 5: Queue length theo thời gian (user + edge)
# =============================================================================

def plot_queue_lengths(ts: pd.DataFrame):
    user_cols = [c for c in ts.columns if c.startswith("u") and c.endswith("_qlen")]
    edge_cols = [c for c in ts.columns if c.startswith("e") and c.endswith("_qlen")]

    if not user_cols and not edge_cols:
        print("[plot] Bỏ qua đồ thị 5: không có cột queue")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=False)

    # --- 5a. User queues ---
    ax = axes[0]
    for col in user_cols:
        uid = col.replace("u", "").replace("_qlen", "")
        ax.plot(ts["sim_time"], rolling_mean(ts[col], 3),
                linewidth=1.2, alpha=0.8, label=f"User {uid}")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Queue length")
    ax.set_title("(a) Hàng đợi User Device")
    ax.legend(ncol=2, fontsize=9)

    # --- 5b. Edge queues ---
    ax = axes[1]
    edge_palette = [C_EDGE0, C_EDGE1, "#E9C46A", "#E76F51"]
    for i, col in enumerate(edge_cols):
        eid = col.replace("e", "").replace("_qlen", "")
        ax.plot(ts["sim_time"], rolling_mean(ts[col], 3),
                color=edge_palette[i % len(edge_palette)],
                linewidth=1.8, label=f"Edge {eid}")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Queue length")
    ax.set_title("(b) Hàng đợi Edge Server")
    ax.legend()

    fig.suptitle("Đồ thị 5 — Queue Length theo thời gian", fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "5_queue_lengths")


# =============================================================================
# Đồ thị 6: Reward heatmap — arm vs app_type
# =============================================================================

def plot_reward_heatmap(task_df: pd.DataFrame):
    needed = {"ucb_arm", "app_name", "ucb_reward"}
    if task_df.empty or not needed.issubset(task_df.columns):
        print("[plot] Bỏ qua đồ thị 6: thiếu cột cần thiết")
        return

    pivot = (task_df.dropna(subset=["ucb_arm", "app_name", "ucb_reward"])
             .groupby(["app_name", "ucb_arm"])["ucb_reward"]
             .mean().unstack(fill_value=np.nan))

    if pivot.empty:
        return

    pivot.index = [APP_LABELS.get(i, i) for i in pivot.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Mean Reward"})
    ax.set_xlabel("Arm (quyết định offload)")
    ax.set_ylabel("Ứng dụng")
    ax.set_title("Đồ thị 6 — Mean UCB Reward theo App × Arm")
    fig.tight_layout()
    savefig(fig, "6_reward_heatmap")


# =============================================================================
# Đồ thị 7: Phân phối latency (histogram + KDE) theo loại xử lý
# =============================================================================

def plot_latency_distribution(task_df: pd.DataFrame):
    if task_df.empty or "latency_ms" not in task_df.columns:
        print("[plot] Bỏ qua đồ thị 7: không có latency_ms")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- 7a. Overall histogram ---
    ax = axes[0]
    valid = task_df["latency_ms"].dropna()
    ax.hist(valid, bins=40, color=C_UCB, alpha=0.75, edgecolor="white")
    ax.axvline(valid.mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Mean = {valid.mean():.1f} ms")
    ax.axvline(valid.quantile(0.95), color="orange", linestyle="--",
               linewidth=1.5, label=f"p95 = {valid.quantile(0.95):.1f} ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Số task")
    ax.set_title("(a) Phân phối Latency tổng thể")
    ax.legend()

    # --- 7b. Theo loại offloading ---
    ax = axes[1]
    groups = {
        "Local":   task_df[task_df["offloaded"] == 0]["latency_ms"].dropna(),
        "Edge":    task_df[(task_df["offloaded"] == 1) & (task_df["split_ratio"] >= 0.99)]["latency_ms"].dropna(),
        "Partial": task_df[(task_df["offloaded"] == 1) & (task_df["split_ratio"] < 0.99)]["latency_ms"].dropna(),
    }
    colors_grp = [C_LOCAL, C_EDGE0, C_EDGE1]
    for (label, data), color in zip(groups.items(), colors_grp):
        if len(data) > 5:
            ax.hist(data, bins=30, alpha=0.55, color=color,
                    edgecolor="white", label=f"{label} (n={len(data)})")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Số task")
    ax.set_title("(b) Phân phối Latency theo loại xử lý")
    ax.legend()

    fig.suptitle("Đồ thị 7 — Phân phối Latency", fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "7_latency_distribution")


# =============================================================================
# Đồ thị 8: Throughput (tasks completed per interval) theo thời gian
# =============================================================================

def plot_throughput(ts: pd.DataFrame):
    if "tasks_done_interval" not in ts.columns:
        print("[plot] Bỏ qua đồ thị 8: không có tasks_done_interval")
        return

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(ts["sim_time"], ts["tasks_done_interval"],
           width=0.9, color=C_UCB, alpha=0.6, label="Tasks / interval")
    ax.plot(ts["sim_time"], rolling_mean(ts["tasks_done_interval"], 5),
            color="red", linewidth=1.8, label="Rolling mean (w=5)")
    ax.set_xlabel("Thời gian mô phỏng (s)")
    ax.set_ylabel("Số task hoàn thành")
    ax.set_title("Đồ thị 8 — Throughput theo thời gian")
    ax.legend()
    fig.tight_layout()
    savefig(fig, "8_throughput")


# =============================================================================
# Đồ thị 9: Edge utilisation (busy flag theo thời gian)
# =============================================================================

def plot_edge_utilisation(ts: pd.DataFrame):
    busy_cols = [c for c in ts.columns if c.endswith("_busy")]
    load_cols = [c for c in ts.columns if c.endswith("_load")]
    if not busy_cols:
        print("[plot] Bỏ qua đồ thị 9: không có cột _busy")
        return

    n = len(busy_cols)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4.5), sharey=False)
    if n == 1:
        axes = [axes]

    edge_colors = [C_EDGE0, C_EDGE1, "#E9C46A", "#E76F51"]
    for i, (bcol, ax) in enumerate(zip(busy_cols, axes)):
        eid = bcol.replace("e", "").replace("_busy", "")
        utilization = rolling_mean(ts[bcol], 5) * 100
        ax.fill_between(ts["sim_time"], 0, utilization,
                        alpha=0.3, color=edge_colors[i])
        ax.plot(ts["sim_time"], utilization,
                color=edge_colors[i], linewidth=1.5)
        ax.set_ylim(0, 110)
        ax.set_xlabel("Thời gian (s)")
        ax.set_ylabel("Utilisation (%)")
        ax.set_title(f"Edge {eid} Utilisation")

        lcol = f"e{eid}_load"
        if lcol in ts.columns:
            ax2 = ax.twinx()
            ax2.plot(ts["sim_time"], rolling_mean(ts[lcol], 5),
                     color="gray", linewidth=1, linestyle=":", alpha=0.7)
            ax2.set_ylabel("Load (cycles)", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")

    fig.suptitle("Đồ thị 9 — Edge Server Utilisation", fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "9_edge_utilisation")


# =============================================================================
# Đồ thị 10: Summary dashboard (1 trang tổng hợp cho báo cáo)
# =============================================================================

def plot_summary_dashboard(ts: pd.DataFrame, task_df: pd.DataFrame, summary: dict):
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # --- Panel A: Reward convergence ---
    ax = fig.add_subplot(gs[0, 0])
    if "ucb_reward_cumulative" in ts.columns:
        valid = ts.dropna(subset=["ucb_reward_cumulative"])
        ax.plot(valid["sim_time"], valid["ucb_reward_cumulative"],
                color=C_REWARD, linewidth=2)
        ax.fill_between(valid["sim_time"], 0, valid["ucb_reward_cumulative"],
                        alpha=0.15, color=C_REWARD)
    ax.set_title("A. Cumulative Reward")
    ax.set_xlabel("Time (s)")

    # --- Panel B: Mean latency ---
    ax = fig.add_subplot(gs[0, 1])
    if "mean_lat_all_ms" in ts.columns:
        valid = ts.dropna(subset=["mean_lat_all_ms"])
        ax.plot(valid["sim_time"], rolling_mean(valid["mean_lat_all_ms"]),
                color=C_UCB, linewidth=2)
    ax.set_title("B. Mean Latency (ms)")
    ax.set_xlabel("Time (s)")

    # --- Panel C: Offload pie ---
    ax = fig.add_subplot(gs[0, 2])
    if not task_df.empty and "ucb_arm" in task_df.columns:
        arm_counts = task_df["ucb_arm"].dropna().value_counts()
        if not arm_counts.empty:
            arm_colors_list = []
            for arm in arm_counts.index:
                if arm == "Local":
                    arm_colors_list.append(C_LOCAL)
                elif "0" in str(arm):
                    arm_colors_list.append(C_EDGE0)
                else:
                    arm_colors_list.append(C_EDGE1)
            ax.pie(arm_counts.values, labels=arm_counts.index,
                   colors=arm_colors_list, autopct="%1.0f%%",
                   startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("C. Offload Decision")

    # --- Panel D: Latency by app ---
    ax = fig.add_subplot(gs[1, 0])
    by_type = summary.get("by_type", {})
    if by_type:
        apps  = list(by_type.keys())
        means = [by_type[a]["latency_ms"].get("mean", 0) for a in apps]
        colors_d = [APP_COLORS.get(a, "#888") for a in apps]
        bars = ax.barh([APP_LABELS.get(a, a) for a in apps], means,
                       color=colors_d, alpha=0.82, edgecolor="white")
        ax.bar_label(bars, fmt="%.0f ms", fontsize=9)
    ax.set_title("D. Mean Latency per App")
    ax.set_xlabel("Latency (ms)")

    # --- Panel E: Throughput ---
    ax = fig.add_subplot(gs[1, 1])
    if "tasks_done_interval" in ts.columns:
        ax.bar(ts["sim_time"], ts["tasks_done_interval"],
               width=0.85, color=C_UCB, alpha=0.55)
        ax.plot(ts["sim_time"], rolling_mean(ts["tasks_done_interval"], 5),
                color="red", linewidth=1.5)
    ax.set_title("E. Throughput (tasks/interval)")
    ax.set_xlabel("Time (s)")

    # --- Panel F: Key metrics text ---
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    lat  = summary.get("latency_all_ms", {})
    dag  = summary.get("dag_completed", "—")
    dag_total = summary.get("dag_total", "—")
    text = (
        f"  Kết quả mô phỏng\n\n"
        f"  Tasks hoàn thành : {summary.get('total_done','—')}\n"
        f"  Offload ratio     : {summary.get('offload_ratio',0)*100:.1f}%\n"
        f"  Mean latency      : {lat.get('mean','—')} ms\n"
        f"  p95 latency       : {lat.get('p95','—')} ms\n"
        f"  DAG completed     : {dag} / {dag_total}\n"
        f"  Policy            : UCB (LinUCB)\n"
    )
    ax.text(0.05, 0.85, text, transform=ax.transAxes,
            fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4ff",
                      edgecolor="#4361EE", linewidth=1.5))

    fig.suptitle("Dashboard — MEC Simulation với UCB Policy",
                 fontsize=15, fontweight="bold")
    savefig(fig, "0_dashboard")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print(" plot.py — Vẽ đồ thị báo cáo MEC Simulation")
    print("=" * 60)

    ts, task_df, summary = load_data()

    print("\n[1/10] Reward convergence...")
    plot_reward_convergence(ts)

    print("[2/10] Latency timeseries...")
    plot_latency_timeseries(ts)

    print("[3/10] Latency by app...")
    plot_latency_by_app(summary)

    print("[4/10] Arm distribution...")
    plot_arm_distribution(task_df)

    print("[5/10] Queue lengths...")
    plot_queue_lengths(ts)

    print("[6/10] Reward heatmap...")
    plot_reward_heatmap(task_df)

    print("[7/10] Latency distribution...")
    plot_latency_distribution(task_df)

    print("[8/10] Throughput...")
    plot_throughput(ts)

    print("[9/10] Edge utilisation...")
    plot_edge_utilisation(ts)

    print("[10/10] Summary dashboard...")
    plot_summary_dashboard(ts, task_df, summary)

    print(f"\nTất cả đồ thị đã lưu vào thư mục: {RESULTS_DIR}/")
    print("   File fig_0_dashboard.png là tổng hợp cho báo cáo.")


if __name__ == "__main__":
    main()