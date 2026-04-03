# =============================================================================
# env/mec_env.py — MEC Environment
# =============================================================================
from __future__ import annotations
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional
from env.task import Task
from env.user_device import UserDevice
from env.edge_server import EdgeServer
from env.channel import SharedBandwidthChannel, FixedChannel
import config as cfg
import pandas as pd
import os

class MecEnv:
    """
    Orchestrator chính.

    Vòng lặp:
        tasks   = env.generate_tasks()
        actions = policy.act(tasks, env.get_obs())
        env.apply_actions(actions)
        env.step()

    Policy trả về action cho mỗi task:
        'local'            → 100% local
        int (edge_id)      → 100% offload lên edge đó
        (edge_id, ratio)   → partial: ratio∈(0,1), ratio% lên edge, còn lại local
    """

    def __init__(self, seed: Optional[int] = None):
        _seed = seed if seed is not None else cfg.RANDOM_SEED
        self._rng = np.random.default_rng(_seed)
        self.users: List[UserDevice] = []
        self.edges: List[EdgeServer] = []
        self.sim_time:       float = 0.0
        self.step_count:     int   = 0
        self.finished_tasks: List[Task] = []
        self._pending:       List[Task] = []
        self.reset()
        
        # ── [NEW] NẠP SẴN 4 MA TRẬN NHIỄU (PRELOAD INTERFERENCE MATRICES) ──
        self.interference_matrices = {}
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        apps = ['lightgbm', 'mapreduce', 'matrix_app', 'video_app']
        
        print("Đang nạp ma trận nhiễu từ profile_data...")
        for app in apps:
            excel_path = os.path.join(base_dir, 'profile_data', f"{app}_mc.xlsx")
            if os.path.exists(excel_path):
                try:
                    m_matrix = pd.read_excel(excel_path, engine='openpyxl', sheet_name='edm').values
                    c_matrix = pd.read_excel(excel_path, engine='openpyxl', sheet_name='edc').values
                    self.interference_matrices[app] = {'m': m_matrix, 'c': c_matrix}
                    print(f" - Đã nạp ma trận cho: {app}")
                except Exception as e:
                    print(f" - Lỗi khi nạp {app}_mc.xlsx: {e}")
            else:
                print(f" - Cảnh báo: Không tìm thấy file {app}_mc.xlsx")
        
        self.reset()

    def reset(self):
        self.sim_time   = 0.0
        self.step_count = 0
        self.finished_tasks.clear()
        self._pending.clear()

        if cfg.CHANNEL_MODEL == 'shared':
            self.channel = SharedBandwidthChannel()
        else:
            self.channel = FixedChannel()

        self.users = [
            UserDevice(i, np.random.default_rng(self._rng.integers(1 << 31)))
            for i in range(cfg.NUM_USERS)
        ]
        self.edges = [EdgeServer(s) for s in cfg.EDGE_SERVERS]

    # ── Main loop ─────────────────────────────────────────────────────────────

    def generate_tasks(self) -> List[Task]:
        all_tasks = []
        for user in self.users:
            all_tasks.extend(user.generate_tasks(cfg.DT, self.sim_time))
        self._pending = all_tasks
        return all_tasks

    def apply_actions(self, actions: List[Tuple[Task, "str | int | tuple"]]):
        """
        Nhận routing từ policy, đẩy vào queue.

        dest = 'local'         → full local
        dest = int             → full edge
        dest = (int, float)    → partial: (edge_id, split_ratio)
                                  split_ratio = phần trăm cycles lên edge
        """
        # Đếm concurrent offload per edge (tính cả partial)
        edge_offload_counts = Counter()
        for _, dest in actions:
            if dest != "local":
                eid = int(dest[0]) if isinstance(dest, tuple) else int(dest)
                edge_offload_counts[eid] += 1

        for task, dest in actions:
            user = self.users[task.user_id]

            if dest == "local":
                self._route_local(task, user)

            elif isinstance(dest, tuple):
                # Partial offloading: (edge_id, split_ratio)
                edge_id, ratio = int(dest[0]), float(dest[1])
                ratio = max(0.01, min(0.99, ratio))  # clamp (0,1)
                self._route_partial(task, user, edge_id, ratio,
                                    edge_offload_counts[edge_id])
            else:
                # Full offload
                edge_id = int(dest)
                self._route_edge(task, user, edge_id,
                                 edge_offload_counts[edge_id])

        self._pending.clear()

    def step(self):
        """Tiến 1 DT, thu thập task hoàn thành."""
        for user in self.users:
            done = user.step(cfg.DT, self.sim_time)
            for t in done:
                self._on_local_part_done(t)

        for edge in self.edges:
            done = edge.step(cfg.DT, self.sim_time)
            for t in done:
                self._on_edge_part_done(t)

        self.sim_time   += cfg.DT
        self.step_count += 1

    # ── Observation ──────────────────────────────────────────────────────────

    def get_obs(self) -> dict:
        return {
            "sim_time": self.sim_time,
            "users":    [u.get_obs() for u in self.users],
            "edges":    [e.get_obs() for e in self.edges],
        }

    @property
    def done(self) -> bool:
        return self.sim_time >= cfg.SIM_DURATION

    @property
    def total_tasks_generated(self) -> int:
        return sum(u.local_count + u.offloaded_count for u in self.users)

    @property
    def offload_ratio(self) -> float:
        total = self.total_tasks_generated
        if total == 0:
            return 0.0
        return sum(u.offloaded_count for u in self.users) / total

    def summary(self) -> dict:
        tasks = self.finished_tasks
        if not tasks:
            return {"total_done": 0}

        # --- Logic tính toán tổng thể (giữ nguyên của bạn) ---
        total_done = len(tasks)
        offload_count = sum(1 for t in tasks if t.offloaded)
        
        def get_stats(lst):
            lats = [t.latency * 1000 for t in lst if not np.isnan(t.latency)]
            if not lats: return {}
            return {
                "mean": round(float(np.mean(lats)), 2),
                "p50":  round(float(np.median(lats)), 2),
                "p95":  round(float(np.percentile(lats, 95)), 2),
                "min":  round(float(np.min(lats)), 2),
                "max":  round(float(np.max(lats)), 2),
            }

        res = {
            "total_done": total_done,
            "total_generated": getattr(self, 'total_generated', total_done),
            "offload_ratio": round(offload_count / total_done, 4) if total_done > 0 else 0,
            "latency_all_ms": get_stats(tasks),
            "latency_local_ms": get_stats([t for t in tasks if not t.offloaded]),
            "latency_edge_ms": get_stats([t for t in tasks if t.offloaded and not t.is_partial]),
            "latency_partial_ms": get_stats([t for t in tasks if t.is_partial]),
        }

        # === [CẬP NHẬT MỚI] PHÂN TÍCH CHI TIẾT TỪNG LOẠI APP ===
        by_type = {}
        # Lấy danh sách các loại app thực tế có trong dữ liệu
        app_types = set(getattr(t, 'app_type', 'Independent') for t in tasks)

        for a_type in app_types:
            # Lọc các task thuộc về loại app này
            app_tasks = [t for t in tasks if getattr(t, 'app_type', 'Independent') == a_type]
            
            # Tính toán thống kê riêng cho App này
            by_type[a_type] = {
                "task_count": len(app_tasks),
                "offload_ratio": round(sum(1 for t in app_tasks if t.offloaded) / len(app_tasks), 4),
                "latency_ms": get_stats(app_tasks)
            }
        
        res["by_type"] = by_type
        return res

    # ── Internal routing ─────────────────────────────────────────────────────

    def _route_local(self, task: Task, user: UserDevice):
        task.offloaded    = False
        task.split_ratio  = 0.0
        task.cycles_local = task.cycles
        task.cycles_edge  = 0.0
        task.queue_start  = self.sim_time
        user.local_count += 1
        user.enqueue(task, self.sim_time)

    def _route_edge(self, task: Task, user: UserDevice,
                    edge_id: int, num_concurrent: int):
        edge     = self.edges[edge_id]
        rate     = self.channel.compute_rate(task.user_id, edge_id, num_concurrent)
        tx_delay = task.input_bits / rate + cfg.PROPAGATION_DELAY

        # ── [NEW] TÍNH TOÁN ĐỘ TRỄ NHIỄU (LINEAR INTERFERENCE MODEL) ──
        base_time = task.cycles / edge.cpu_freq  # Thời gian gốc khi chạy 1 mình
        penalty_time = 0.0                       # Thời gian phạt do chạy nền
        
        # Lấy tên app từ task (bạn có thể lưu thuộc tính này lúc parse DAG)
        app_name = getattr(task, 'app_type', None) 
        
        if app_name and app_name in self.interference_matrices:
            m_matrix = self.interference_matrices[app_name]['m']
            c_matrix = self.interference_matrices[app_name]['c']
            
            # Lấy trung bình hệ số m và c của thiết bị edge_id
            m_factor = abs(m_matrix[edge_id % len(m_matrix)].mean())
            c_factor = abs(c_matrix[edge_id % len(c_matrix)].mean())
            
            base_time = c_factor
            penalty_time = m_factor * num_concurrent  # Phương trình tuyến tính m*x + c
        else:
            # Fallback: Nếu task không có app_name, dùng hệ số phạt mặc định 50ms
            penalty_time = 0.05 * num_concurrent 
            
        predicted_exec_time = base_time + penalty_time
        
        # QUAN TRỌNG: Quy đổi ngược thời gian bị nhiễu thành "Cycles Ảo" để đánh lừa CPU Queue
        adjusted_cycles = predicted_exec_time * edge.cpu_freq
        # ──────────────────────────────────────────────────────────────

        task.offloaded        = True
        task.split_ratio      = 1.0
        task.edge_id          = edge_id
        task.tx_delay         = tx_delay
        task.channel_rate     = rate
        task.cycles_local     = 0.0
        task.cycles_edge      = adjusted_cycles  # Sử dụng số Cycles ảo đã tính thêm nhiễu
        user.offloaded_count += 1
        edge.receive_offloaded_task(task, self.sim_time)

    def _route_partial(self, task: Task, user: UserDevice,
                       edge_id: int, ratio: float, num_concurrent: int):
        """
        Chia task: ratio% lên edge, (1-ratio)% ở local, chạy song song.
        Latency = max(finish_local, finish_edge) - arrival_time
        """
        edge     = self.edges[edge_id]
        rate     = self.channel.compute_rate(task.user_id, edge_id, num_concurrent)
        # Chỉ truyền phần data tỉ lệ với ratio
        tx_delay = (task.input_bits * ratio) / rate + cfg.PROPAGATION_DELAY

        task.offloaded        = True
        task.split_ratio      = ratio
        task.edge_id          = edge_id
        task.tx_delay         = tx_delay
        task.channel_rate     = rate
        task.cycles_local     = task.cycles * (1 - ratio)
        task.cycles_edge      = task.cycles * ratio
        task.queue_start      = self.sim_time   # local part bắt đầu ngay
        user.offloaded_count += 1

        # Local part → user queue
        user.enqueue_partial(task, self.sim_time)
        # Edge part → edge tx_buffer
        edge.receive_offloaded_task(task, self.sim_time)

    # ── Completion handlers ──────────────────────────────────────────────────

    def _on_local_part_done(self, task: Task):
        """Gọi khi user CPU xử lý xong local part."""
        task.done_local = True
        
        if task.is_partial:
            # Xử lý task bị cắt đôi (Partial)
            if task.done_edge and not task.done:
                task.done = True
                self.finished_tasks.append(task)
                if task.job_id == 1:
                    print(f"[DAG {task.app_type}] Task {task.dag_name} làm xong (Partial) lúc {self.sim_time:.3f}s")
                    
        elif not task.offloaded:
            # [FIX 1] Chỉ đếm nếu đây là task thuần Local
            if not task.done:
                # [FIX 2] Bỏ qua các Dummy Node (có cycles = 0) không đi qua Router
                if task.cycles > 0 or hasattr(task, 'queue_start'):
                    task.done = True
                    self.finished_tasks.append(task)
                    if task.job_id == 1:
                        print(f"[DAG {task.app_type}] Task {task.dag_name} làm xong (Local) lúc {self.sim_time:.3f}s")

    def _on_edge_part_done(self, task: Task):
        """Gọi khi edge CPU xử lý xong edge part."""
        task.done_edge = True
        
        if task.is_partial:
            # Xử lý task bị cắt đôi (Partial)
            if task.done_local and not task.done:
                task.done = True
                self.finished_tasks.append(task)
                if task.job_id == 1:
                    print(f"[DAG {task.app_type}] Task {task.dag_name} làm xong (Partial) lúc {self.sim_time:.3f}s")
                    
        elif task.offloaded:
            # [FIX 3] Chỉ đếm nếu đây là task thuần Edge (Full Offload)
            if not task.done:
                task.done = True
                self.finished_tasks.append(task)
                if task.job_id == 1:
                    print(f"[DAG {task.app_type}] Task {task.dag_name} làm xong (Edge) lúc {self.sim_time:.3f}s")

    def _stats_by_type(self, tasks):
        result = {}
        for tt in cfg.TASK_TYPES:
            name = tt["name"]
            sub  = [t for t in tasks if t.task_type == name
                    and not np.isnan(t.latency)]
            if not sub:
                continue
            lats = np.array([t.latency for t in sub]) * 1000
            result[name] = {
                "count":       len(sub),
                "mean_ms":     round(float(np.mean(lats)),           2),
                "p95_ms":      round(float(np.percentile(lats, 95)), 2),
                "offload_pct": round(sum(1 for t in sub if t.offloaded)
                                     / len(sub) * 100, 1),
                "partial_pct": round(sum(1 for t in sub if t.is_partial)
                                     / len(sub) * 100, 1),
            }
        return result