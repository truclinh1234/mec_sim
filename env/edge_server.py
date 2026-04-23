# =============================================================================
# env/edge_server.py — Edge Server
# =============================================================================
from __future__ import annotations
from collections import deque
from typing import List
from env.node import QueueNode
from env.task import Task
import config as cfg


class EdgeServer(QueueNode):
    """
    Edge Server với 2 buffer tách biệt:

        tx_buffer   — task đang truyền qua kênh (chưa vào RAM/CPU)
        queue (CPU) — task đã truyền xong, chờ CPU xử lý

    Timing rõ ràng:
        tx_start_time  — lúc bắt đầu truyền lên kênh
        queue_start    — lúc thực sự vào hàng đợi CPU (sau khi truyền xong)
        proc_start     — lúc CPU bắt đầu xử lý

    Hai loại delay:
        Transmission delay = tx_start_time + tx_delay  (bay trên kênh)
        Queueing delay     = proc_start - queue_start  (chờ CPU)
    """

    def __init__(self, server_cfg: dict):
        super().__init__(
            node_id        = server_cfg["id"],
            cpu_freq       = server_cfg["cpu_freq"],
            queue_capacity = server_cfg.get("queue_capacity", 0),
            label          = server_cfg.get("label", f"Edge-{server_cfg['id']}"),
        )
        self.tx_buffer: deque[Task] = deque()

    # ── Public API ────────────────────────────────────────────────────────────

    def receive_offloaded_task(self, task: Task, current_time: float) -> bool:
        """
        Nhận task offload từ User.
        Task vào tx_buffer — đang bay trên kênh, chưa chạm CPU queue.
        """
        task.tx_start_time = current_time   # lúc bắt đầu truyền
        self.tx_buffer.append(task)
        return True

    def step(self, dt: float, current_time: float) -> List[Task]:
        """
        Mỗi bước:
          1. Flush tx_buffer → cpu queue (task truyền xong)
          2. While loop vắt kiệt cycles trong dt:
             task xong sớm → dùng cycles dư chạy task tiếp luôn
        """
        finished: List[Task] = []

        # ── Bước 1: flush tx_buffer → cpu queue ──────────────────────────────
        still_tx = deque()
        for task in self.tx_buffer:
            if task.tx_start_time + task.tx_delay <= current_time:
                if self.queue_capacity == 0 or len(self.queue) < self.queue_capacity:
                    task.queue_start = current_time  # thực sự vào hàng đợi CPU
                    self.queue.append(task)
                else:
                    self.total_dropped += 1          # CPU queue đầy mới drop
            else:
                still_tx.append(task)
        self.tx_buffer = still_tx

        # ── Bước 2: while loop — 
        cycles_left = self.cpu_freq * dt

        while cycles_left > 0:
            if self.proc is None:
                if self.queue:
                    task = self.queue.popleft()
                    task.edge_proc_start = current_time
                    self.proc = task
                    # Partial task chỉ xử lý cycles_edge
                    self.rem = task.cycles_edge if task.is_partial else task.cycles
                else:
                    break

            if self.rem <= cycles_left:
                cycles_left -= self.rem
                finish = current_time + (self.cpu_freq * dt - cycles_left) / self.cpu_freq
                # Luôn ghi finish_time_edge cho cả full edge lẫn partial
                self.proc.finish_time_edge = finish
                if not self.proc.is_partial:
                    self.proc.finish_time = finish
                finished.append(self.proc)
                self.total_done += 1
                self.proc = None
                self.rem  = 0.0
            else:
                self.rem    -= cycles_left
                cycles_left  = 0

        return finished

    # ── Observation ──────────────────────────────────────────────────────────

    @property
    def load(self) -> int:
        """Task đang xử lý + chờ CPU (không tính tx_buffer)."""
        return len(self.queue) + (1 if self.is_busy else 0)

    @property
    def total_load(self) -> int:
        """Tổng task kể cả đang truyền trên kênh."""
        return self.load + len(self.tx_buffer)

    def get_obs(self) -> dict:
        return {
            "edge_id":    self.node_id,
            "label":      self.label,
            "queue_len":  self.queue_len,
            "tx_buffer":  len(self.tx_buffer),
            "is_busy":    self.is_busy,
            "load":       self.load,
            "total_load": self.total_load,
            "cpu_freq":   self.cpu_freq,
            "total_done": self.total_done,
        }

    def reset(self):
        super().reset()
        self.tx_buffer.clear()