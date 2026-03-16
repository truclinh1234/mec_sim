# config.py — Tất cả tham số hệ thống

NUM_USERS = 6

# Mỗi dict là 1 edge server: id, cpu_freq (Hz), queue_capacity (0 = unlimited)
EDGE_SERVERS = [
    {"id": 0, "cpu_freq": 3e9, "queue_capacity": 0, "label": "Edge-1 (3GHz)"},
    {"id": 1, "cpu_freq": 2e9, "queue_capacity": 0, "label": "Edge-2 (2GHz)"},
]

# CPU tần số của user device (Hz)
USER_CPU_FREQ = 1e9  # 1 GHz

# TASK GENERATION

# Tốc độ đến trung bình (tasks/giây) cho mỗi user — phân phối Poisson
ARRIVAL_RATE = 2.0

# Định nghĩa loại task
# cycles: số chu kỳ CPU cần để xử lý
# input_bits: kích thước dữ liệu cần truyền khi offload
# prob: xác suất xuất hiện (tổng = 1.0)
TASK_TYPES = [
    {"name": "Light",  "cycles": 3e8, "input_bits": 1e6,  "prob": 0.50},
    {"name": "Medium", "cycles": 1e9, "input_bits": 5e6,  "prob": 0.35},
    {"name": "Heavy",  "cycles": 2e9, "input_bits": 20e6, "prob": 0.15},
]

# ──────────────────────────────────────────────────────────────────────────────
# NETWORK
# ──────────────────────────────────────────────────────────────────────────────

# Tốc độ kênh truyền từ user lên edge (bits/s)
CHANNEL_RATE_BPS = 10e6  # 10 Mbps — tổng băng thông kênh

# Channel model: 'shared' | 'fixed'
#   shared — chia đều băng thông cho số user offload cùng lúc (thực tế hơn)
#   fixed  — mỗi user luôn được full CHANNEL_RATE_BPS (baseline)
CHANNEL_MODEL = 'shared' 

# Trễ cố định mỗi lần offload (s) — RTT overhead, protocol handshake
PROPAGATION_DELAY = 0.002  # 2 ms

# ──────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────────────────────

SIM_DURATION = 60.0    # Tổng thời gian mô phỏng (giây)
DT           = 0.01    # Time step (giây) — nhỏ hơn = chính xác hơn, chậm hơn
RANDOM_SEED  = 42      # Seed cho tái lập kết quả

# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

METRICS_INTERVAL = 1.0          # Lưu metrics mỗi N giây
METRICS_OUTPUT_DIR = "results"  # Thư mục xuất file CSV / JSON

# ──────────────────────────────────────────────────────────────────────────────
# POLICY DEFAULTS (dùng khi khởi tạo từ config, có thể override trong code)
# ──────────────────────────────────────────────────────────────────────────────

# ThresholdPolicy
THRESHOLD_QUEUE_LEN = 2        # Offload khi queue >= giá trị này
THRESHOLD_HEAVY_ALWAYS = True  # Heavy task luôn offload bất kể queue

# RandomPolicy
RANDOM_OFFLOAD_PROB = 0.5      # Xác suất offload mỗi task