# =============================================================================
# env/node.py — M/D/1 Queue Node (dùng chung cho User Device & Edge Server)
# =============================================================================
from __future__ import annotations
from collections import deque
from typing import Optional, List
from env.task import Task


class QueueNode:
    """
    Mô hình hàng đợi CPU đơn giản (M/D/1-style).

    Arrival:       Poisson (xử lý ở MecEnv / UserDevice)
    Service time:  Deterministic — service_time = task.cycles / cpu_freq
    Servers:       1 CPU (single server)
    Queue:         FCFS, capacity tuỳ chỉnh (0 = vô hạn)

    State mỗi bước:
        - proc:   Task đang chạy (None nếu idle)
        - rem:    CPU cycles còn lại của proc
        - queue:  deque các task chờ
    """

    def __init__(
        self,
        node_id: int,
        cpu_freq: float,     # Hz
        queue_capacity: int = 0,   # 0 = unlimited
        label: str = "",
    ):
        self.node_id        = node_id
        self.cpu_freq       = cpu_freq
        self.queue_capacity = queue_capacity
        self.label          = label or f"Node-{node_id}"

        self.queue: deque[Task] = deque()
        self.proc:  Optional[Task] = None
        self.rem:   float = 0.0    # cycles còn lại của task đang xử lý

        # Thống kê nội bộ (reset mỗi khi reset env)
        self.total_done:    int   = 0
        self.total_dropped: int   = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def enqueue(self, task: Task, current_time: float) -> bool:
        """
        Thêm task vào queue.
        Trả về True nếu thành công, False nếu queue đầy (dropped).
        """
        if self.queue_capacity > 0 and len(self.queue) >= self.queue_capacity:
            self.total_dropped += 1
            return False
        task.queue_start = current_time
        self.queue.append(task)
        return True

    def step(self, dt: float, current_time: float) -> List[Task]:
        """
        Tiến time step dt giây.
        Dùng while loop vắt kiệt cycles trong dt:
        task xong sớm → dùng cycles dư chạy task tiếp luôn, không lãng phí.
        """
        finished: List[Task] = []
        cycles_left = self.cpu_freq * dt

        while cycles_left > 0:
            if self.proc is None:
                if self.queue:
                    self._start_next(current_time)
                else:
                    break  # Hết task → CPU nghỉ

            if self.rem <= cycles_left:
                # Task xong trong bước này, còn dư cycles
                cycles_left -= self.rem
                self.proc.finish_time = current_time + (self.cpu_freq * dt - cycles_left) / self.cpu_freq
                self.proc.done = True
                finished.append(self.proc)
                self.total_done += 1
                self.proc = None
                self.rem  = 0.0
                # while tiếp tục → dùng cycles_left cho task tiếp theo
            else:
                # Task chưa xong, tiêu hết cycles
                self.rem    -= cycles_left
                cycles_left  = 0

        return finished

    # ── Observation helpers ──────────────────────────────────────────────────

    @property
    def queue_len(self) -> int:
        return len(self.queue)

    @property
    def is_busy(self) -> bool:
        return self.proc is not None

    @property
    def utilization_approx(self) -> float:
        """
        Ước tính utilization = 1 nếu đang xử lý, 0 nếu idle.
        Dùng để đưa vào observation của policy.
        """
        return 1.0 if self.is_busy else 0.0

    @property
    def load(self) -> int:
        """Số task đang & chờ xử lý (dùng để so sánh load giữa các edge)."""
        return self.queue_len + (1 if self.is_busy else 0)

    def snapshot(self) -> dict:
        """Trả về dict mô tả trạng thái hiện tại (cho metrics)."""
        return {
            "node_id":    self.node_id,
            "label":      self.label,
            "queue_len":  self.queue_len,
            "is_busy":    self.is_busy,
            "total_done": self.total_done,
            "cpu_freq":   self.cpu_freq,
        }

    def reset(self):
        self.queue.clear()
        self.proc  = None
        self.rem   = 0.0
        self.total_done    = 0
        self.total_dropped = 0

    # ── Internal ─────────────────────────────────────────────────────────────

    def _start_next(self, current_time: float):
            task = self.queue.popleft()
            task.proc_start = current_time
            self.proc = task
            
            if "Edge" in self.label:
                # Nếu CPU này là Edge, lấy cycles_edge (đã bao gồm Partial ratio và Nhiễu)
                self.rem = task.cycles_edge
            else:
                # Nếu CPU này là Local, lấy cycles_local
                self.rem = task.cycles_local