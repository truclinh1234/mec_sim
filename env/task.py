# =============================================================================
# env/task.py — Task model (Updated for DAG Support)
# =============================================================================
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class Task:
    """
    1 computation task. Hỗ trợ 3 chế độ: Full local, Full edge, Partial.
    Đã được nâng cấp để hỗ trợ DAG.
    """

    task_id:    int
    user_id:    int
    task_type:  str    # 'Light' | 'Medium' | 'Heavy' | 'DAG_Task'
    cycles:     float  # Tương ứng với 'file' size trong JSON (độ phức tạp tính toán)
    input_bits: float  # Tổng dung lượng input từ các predecessor
    arrival_time: float

    inter_arrival_gap: Optional[float] = None

    # ── DAG Routing & Dependency ──────────────────────────────────────────────
    job_id:     Optional[int] = None    # Thuộc về DAG Job nào
    dag_name:   Optional[str] = None    # Tên vertex trong JSON (vd: "0", "1")
    app_type: Optional[str] = None # 'lightgbm' | 'mapreduce' | 'matrix' | 'video'
    model_size: float = 0.0             # Kích thước model cần tải (từ JSON)
    
    # Danh sách các task_id đi trước (cần xong trước khi task này chạy)
    predecessors: List[int] = field(default_factory=list) 
    # Danh sách các task_id đi sau
    successors:   List[int] = field(default_factory=list) 
    
    ready_to_start: bool = True # Với task thường là True, task DAG thì False cho đến khi preds xong

    # ── Routing ──────────────────────────────────────────────────────────────
    offloaded:   bool  = False
    edge_id:     Optional[int] = None
    split_ratio: float = 0.0   

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
    finish_time:    float = 0.0   
    done_local:     bool  = False

    # ── Timing edge part ─────────────────────────────────────────────────────
    edge_queue_start: float = 0.0
    edge_proc_start:  float = 0.0
    finish_time_edge: float = 0.0
    done_edge:        bool  = False

    # ── Trạng thái hoàn thành ─────────────────────────────────────────────────
    done: bool = False   

    # ── Derived metrics───────────────────────────────────

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
        if not self.done or self.offloaded and not self.is_partial:
            return float("nan")
        return self.queue_start - self.arrival_time

    @property
    def waiting_time(self) -> float:
        if not self.done or self.offloaded and not self.is_partial:
            return float("nan")
        return self.proc_start - self.queue_start

    @property
    def processing_time(self) -> float:
        if not self.done:
            return float("nan")
        if self.offloaded and not self.is_partial:
            return self.finish_time_edge - self.edge_proc_start
        return self.finish_time - self.proc_start

    @property
    def edge_latency(self) -> float:
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
        dag_info = f" DAG:{self.dag_name}" if self.job_id is not None else ""
        return f"Task(id={self.task_id}{dag_info} uid={self.user_id} {self.task_type} {mode} {status})"