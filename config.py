# config.py — Tất cả tham số hệ thống

NUM_USERS = 6

# Mỗi dict là 1 edge server: id, cpu_freq (Hz), queue_capacity (0 = unlimited)
EDGE_SERVERS = [
    {"id": 0, "cpu_freq": 3e9, "queue_capacity": 0, "label": "Edge-1 (3GHz)"},
    {"id": 1, "cpu_freq": 3e9, "queue_capacity": 0, "label": "Edge-2 (3GHz)"},
]

# CPU tần số của user device (Hz)
USER_CPU_FREQ = 1e9  # 1 GHz

# ── TASK GENERATION ───────────────────────────────────────────────────────────
# Tốc độ đến trung bình (jobs/giây) — phân phối Poisson
# Sau khi tích hợp DAG: mỗi "arrival" là 1 DAGJob (không phải task đơn lẻ)
ARRIVAL_RATE = 2.0

# Định nghĩa loại task (giữ để tương thích với các phần còn lại của mec_sim)
TASK_TYPES = [
    {"name": "Light",  "cycles": 3e8, "input_bits": 1e6,  "prob": 0.50},
    {"name": "Medium", "cycles": 1e9, "input_bits": 5e6,  "prob": 0.35},
    {"name": "Heavy",  "cycles": 2e9, "input_bits": 20e6, "prob": 0.15},
]

# ── NETWORK ───────────────────────────────────────────────────────────────────
# Tốc độ kênh truyền từ user lên edge (bits/s)
CHANNEL_RATE_BPS = 10e6  # 10 Mbps — tổng băng thông kênh

# Channel model: 'shared' | 'fixed'
#   shared — chia đều băng thông cho số user offload cùng lúc (thực tế hơn)
#   fixed  — mỗi user luôn được full CHANNEL_RATE_BPS (baseline)
CHANNEL_MODEL = 'shared'

# Trễ cố định mỗi lần offload (s) — RTT overhead, protocol handshake
PROPAGATION_DELAY = 0.002  # 2 ms

# ── SIMULATION ────────────────────────────────────────────────────────────────
SIM_DURATION = 60.0    # Tổng thời gian mô phỏng (giây)
DT           = 0.01    # Time step (giây)
RANDOM_SEED  = 42      # Seed cho tái lập kết quả

# ── METRICS ───────────────────────────────────────────────────────────────────
METRICS_INTERVAL   = 1.0        # Lưu metrics mỗi N giây
METRICS_OUTPUT_DIR = "results"  # Thư mục xuất file CSV / JSON

# ── POLICY DEFAULTS ───────────────────────────────────────────────────────────
# ThresholdPolicy
THRESHOLD_QUEUE_LEN    = 2     # Offload khi queue >= giá trị này
THRESHOLD_HEAVY_ALWAYS = True  # Heavy task luôn offload bất kể queue
# RandomPolicy
RANDOM_OFFLOAD_PROB = 0.5      # Xác suất offload mỗi task

# ── DAG INTEGRATION (IBDASH trace) ────────────────────────────────────────────
# Danh sách app_config.json — mỗi lần spawn DAGJob sẽ random chọn 1 app
DAG_APPS = [
    "profile_data/lightgbm/app_config.json",
    "profile_data/mapreduce/app_config.json",
    "profile_data/matrix_app/app_config.json",
    "profile_data/video_app/app_config.json",
]

# Mapping size → mec_sim task type
# Lưu ý: dag_job.py đã tự dùng hàm nội bộ _size_to_mec() với logic giống hệt.
# Hàm này giữ lại ở đây để:
#   (a) tham khảo / document hoá logic mapping
#   (b) override nếu muốn dùng EDx.csv thay cho ước tính từ size
def dag_task_to_mec_type(data_size_kb: float, model_size_kb: float) -> dict:
    """
    Convert (data_size_kb, model_size_kb) → cycles + input_bits cho mec_sim.

    Mapping:
      total_kb == 0         → Dummy  : cycles=0
      total_kb <  100       → Light  : cycles=3e8
      100 ≤ total_kb < 1000 → Medium : cycles=1e9
      total_kb ≥ 1000       → Heavy  : cycles=2e9

    Nếu sau này muốn dùng execution time thực đo từ EDx.csv,
    thay phần thân hàm này — dag_job.py sẽ tự dùng theo.
    """
    total_kb = data_size_kb + model_size_kb
    if total_kb == 0:
        return {"type": "Light", "cycles": 3e8, "input_bits": 0}
    elif total_kb < 100:
        return {"type": "Light",  "cycles": 3e8, "input_bits": total_kb * 8e3}
    elif total_kb < 1000:
        return {"type": "Medium", "cycles": 1e9, "input_bits": total_kb * 8e3}
    else:
        return {"type": "Heavy",  "cycles": 2e9, "input_bits": total_kb * 8e3}