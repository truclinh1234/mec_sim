# =============================================================================
# plot.py — Phiên bản So sánh Đa tham số (A/B Testing)
# =============================================================================
import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "figure.dpi":       100,
    "savefig.dpi":      300,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

RESULTS_DIR = "results"

# Cấu hình 5 policy với màu sắc khác nhau
POLICIES = {
    "UCB_alpha_0.1": {"color": "#90E0EF", "label": "LinUCB (\u03b1=0.1) - Ít khám phá"},
    "UCB_alpha_1.0": {"color": "#0077B6", "label": "LinUCB (\u03b1=1.0) - Cân bằng"},
    "UCB_alpha_2.5": {"color": "#03045E", "label": "LinUCB (\u03b1=2.5) - Rất tò mò"},
    "Eps_0.05":      {"color": "#F4A261", "label": "\u03b5-Greedy (5% random)"},
    "Eps_0.20":      {"color": "#E76F51", "label": "\u03b5-Greedy (20% random)"}
}

APP_LABELS = {
    "lightgbm":  "LightGBM",
    "mapreduce": "MapReduce",
    "matrix":    "Matrix App",
    "video":     "Video App",
}

def load_all_data():
    ts_data, task_data, summary_data = {}, {}, {}
    for p in POLICIES.keys():
        ts_path  = os.path.join(RESULTS_DIR, f"{p}_timeseries.csv")
        tsk_path = os.path.join(RESULTS_DIR, f"{p}_tasks.csv")
        sum_path = os.path.join(RESULTS_DIR, f"{p}_summary.json")
        
        if os.path.exists(ts_path):
            ts_data[p] = pd.read_csv(ts_path)
            if os.path.exists(tsk_path):
                task_data[p] = pd.read_csv(tsk_path)
            if os.path.exists(sum_path):
                with open(sum_path) as f:
                    summary_data[p] = json.load(f)
    return ts_data, task_data, summary_data

def savefig(fig, name: str):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  → Đã lưu: {path}")
    plt.close(fig)

def rolling_mean(series: pd.Series, w: int = 5) -> pd.Series:
    return series.rolling(w, min_periods=1).mean()

# =============================================================================
# 1. Đồ thị: Quá trình học tập (Reward & Regret)
# =============================================================================
def plot_learning_curve(ts_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- 1a. Cumulative Reward (Bên trái) ---
    ax = axes[0]
    for p, df in ts_data.items():
        if "ucb_reward_cumulative" in df.columns:
            valid = df.dropna(subset=["ucb_reward_cumulative"])
            color = POLICIES[p]["color"]
            label = POLICIES[p]["label"]
            
            ax.plot(valid["sim_time"], valid["ucb_reward_cumulative"], 
                    color=color, linewidth=2.5, label=label)
            ax.fill_between(valid["sim_time"], 0, valid["ucb_reward_cumulative"], 
                            alpha=0.1, color=color)

    ax.set_title("Cumulative Reward\n[Mục tiêu: Càng cao càng tốt]", fontweight="bold")
    ax.set_xlabel("Thời gian mô phỏng (s)")
    ax.set_ylabel("Tổng phần thưởng")
    ax.legend()

    # --- 1b. Cumulative Regret (Bên phải) ---
    ax = axes[1]
    for p, df in ts_data.items():
        if "ucb_regret_cumulative" in df.columns:
            valid = df.dropna(subset=["ucb_regret_cumulative"])
            color = POLICIES[p]["color"]
            label = POLICIES[p]["label"]
            
            ax.plot(valid["sim_time"], valid["ucb_regret_cumulative"], 
                    color=color, linewidth=2.5, linestyle="--", label=label)

    ax.set_title("Cumulative Regret\n[Mục tiêu: Càng thấp & Càng phẳng càng tốt]", fontweight="bold")
    ax.set_xlabel("Thời gian mô phỏng (s)")
    ax.set_ylabel("Tổng hối tiếc (Khoảng cách tới mức tối ưu)")
    ax.legend()

    fig.tight_layout()
    savefig(fig, "1_Compare_Learning_Curve")

# --- Đồ thị 2: Latency ---
def plot_latency_comparison(ts_data):
    fig, ax = plt.subplots(figsize=(10, 5))
    for p, df in ts_data.items():
        if "mean_lat_all_ms" in df.columns:
            valid = df.dropna(subset=["mean_lat_all_ms"])
            smoothed = rolling_mean(valid["mean_lat_all_ms"], w=5)
            ax.plot(valid["sim_time"], smoothed, 
                    color=POLICIES[p]["color"], linewidth=2, label=POLICIES[p]["label"])
    ax.set_title("Độ trễ Hệ thống theo thời gian (Rolling Mean)", fontweight="bold")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Độ trễ (ms)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    savefig(fig, "2_Compare_Params_Latency")

# --- Đồ thị 3: Bar chart P95 theo App (Auto-offset cho 5 cột) ---
def plot_dag_latency_comparison(summary_data):
    if not summary_data: return
    apps = list(list(summary_data.values())[0].get("by_type", {}).keys())
    if not apps: return

    x = np.arange(len(apps))
    n_policies = len(POLICIES)
    total_width = 0.8
    w = total_width / n_policies
    offsets = np.linspace(-total_width/2 + w/2, total_width/2 - w/2, n_policies)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, p in enumerate(POLICIES.keys()):
        if p in summary_data:
            p95s = [summary_data[p]["by_type"][a]["latency_ms"].get("p95", 0) for a in apps]
            ax.bar(x + offsets[i], p95s, w, label=POLICIES[p]["label"], 
                   color=POLICIES[p]["color"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([APP_LABELS.get(a, a) for a in apps])
    ax.set_ylabel("P95 Latency (ms)")
    ax.set_title("Độ trễ P95 theo Ứng dụng", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    savefig(fig, "3_Compare_Params_P95_Latency")

# --- Đồ thị 4: Action Distribution ---
def plot_action_distribution(task_data):
    labels = ["Local", "Edge 0", "Edge 1"]
    dist_data = {}
    
    for p, df in task_data.items():
        if "ucb_arm" in df.columns:
            counts = df["ucb_arm"].dropna().value_counts()
            local_c = counts.get("Local", 0)
            e0_c = counts.get("0", 0) + counts.get("Edge 0", 0)
            e1_c = counts.get("1", 0) + counts.get("Edge 1", 0)
            total = local_c + e0_c + e1_c
            if total > 0:
                dist_data[p] = [(local_c/total)*100, (e0_c/total)*100, (e1_c/total)*100]

    if not dist_data: return

    x = np.arange(len(labels))
    n_policies = len(POLICIES)
    total_width = 0.8
    w = total_width / n_policies
    offsets = np.linspace(-total_width/2 + w/2, total_width/2 - w/2, n_policies)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in enumerate(POLICIES.keys()):
        if p in dist_data:
            ax.bar(x + offsets[i], dist_data[p], w, label=POLICIES[p]["label"],
                   color=POLICIES[p]["color"], edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Tỉ lệ ra quyết định (%)")
    ax.set_title("Chiến lược Phân bổ Tài nguyên", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    savefig(fig, "4_Compare_Params_Action_Dist")

def main():
    print("=" * 60)
    print(" plot.py — Tạo Đồ thị Đa Tham số (5 Variants)")
    print("=" * 60)
    ts_data, task_data, summary_data = load_all_data()
    
    if not ts_data:
        print("[Lỗi] Không có dữ liệu. Chạy python main.py trước.")
        return

    plot_learning_curve(ts_data)
    plot_latency_comparison(ts_data)
    plot_dag_latency_comparison(summary_data)
    plot_action_distribution(task_data)
    print(f"\nHoàn tất! Kiểm tra file ảnh trong thư mục {RESULTS_DIR}/")

if __name__ == "__main__":
    main()