# =============================================================================
# env/task.py — Task model
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Task:
    """
    1 computation task. Hỗ trợ 3 chế độ:
      - Full local  : cycles_edge=0,  split_ratio=0.0
      - Full edge   : cycles_local=0, split_ratio=1.0
      - Partial     : cả 2 > 0,       0 < split_ratio < 1
                      local và edge chạy SONG SONG
                      latency = max(finish_local, finish_edge) - arrival_time

    Bất biến:
      cycles_local + cycles_edge = cycles
      latency = transit_time + waiting_time + processing_time  (full local/edge)
      latency = max(finish_local, finish_edge) - arrival_time  (partial)
    """

    task_id:    int
    user_id:    int
    task_type:  str    # 'Light' | 'Medium' | 'Heavy'
    cycles:     float
    input_bits: float
    arrival_time: float

    inter_arrival_gap: Optional[float] = None

    # ── Routing ──────────────────────────────────────────────────────────────
    offloaded:   bool  = False
    edge_id:     Optional[int] = None
    split_ratio: float = 0.0   # 0.0=full local, 1.0=full edge, (0,1)=partial

    # ── Workload split ───────────────────────────────────────────────────────
    cycles_local: float = 0.0
    cycles_edge:  float = 0.0

    # ── Timing chung ─────────────────────────────────────────────────────────
    tx_start_time:  float = 0.0
    tx_delay:       float = 0.0
    channel_rate:   float = 0.0

    # ── Timing local part ────────────────────────────────────────────────────
    queue_start:    float = 0.0
    proc_start:     float = 0.0
    finish_time:    float = 0.0   # finish của local part (hoặc duy nhất nếu full)
    done_local:     bool  = False

    # ── Timing edge part (chỉ dùng khi partial hoặc full edge) ──────────────
    edge_queue_start: float = 0.0
    edge_proc_start:  float = 0.0
    finish_time_edge: float = 0.0
    done_edge:        bool  = False

    # ── Trạng thái hoàn thành ─────────────────────────────────────────────────
    done: bool = False   # True khi CẢ 2 part xong (hoặc part duy nhất xong)

    # ── Derived metrics ──────────────────────────────────────────────────────

    @property
    def is_partial(self) -> bool:
        return self.cycles_local > 0 and self.cycles_edge > 0

    @property
    def latency(self) -> float:
        if not self.done:
            return float("nan")
        if self.is_partial:
            return max(self.finish_time, self.finish_time_edge) - self.arrival_time
        elif self.offloaded:
            return self.finish_time_edge - self.arrival_time
        else:
            return self.finish_time - self.arrival_time

    @property
    def transit_time(self) -> float:
        """Thời gian từ arrival đến khi vào CPU queue (local part)."""
        if not self.done or self.offloaded and not self.is_partial:
            return float("nan")
        return self.queue_start - self.arrival_time

    @property
    def waiting_time(self) -> float:
        """Chờ trong CPU queue (local part)."""
        if not self.done or self.offloaded and not self.is_partial:
            return float("nan")
        return self.proc_start - self.queue_start

    @property
    def processing_time(self) -> float:
        """CPU xử lý thuần."""
        if not self.done:
            return float("nan")
        if self.offloaded and not self.is_partial:
            # Full edge: tính từ edge_proc_start đến finish_time_edge
            return self.finish_time_edge - self.edge_proc_start
        return self.finish_time - self.proc_start

    @property
    def edge_latency(self) -> float:
        """Latency của edge part (tx + wait + proc trên edge)."""
        if not self.done or not self.offloaded:
            return float("nan")
        return self.finish_time_edge - self.arrival_time

    @property
    def local_ratio(self) -> float:
        return self.cycles_local / self.cycles if self.cycles > 0 else 0.0

    @property
    def edge_ratio(self) -> float:
        return self.cycles_edge / self.cycles if self.cycles > 0 else 0.0

    def __repr__(self) -> str:
        if self.is_partial:
            mode = f"partial({self.local_ratio*100:.0f}%L+{self.edge_ratio*100:.0f}%E→e{self.edge_id})"
        elif self.offloaded:
            mode = f"edge{self.edge_id}"
        else:
            mode = "local"
        status = f"done lat={self.latency*1000:.1f}ms" if self.done else "pending"
        return f"Task(id={self.task_id} uid={self.user_id} {self.task_type} {mode} {status})"