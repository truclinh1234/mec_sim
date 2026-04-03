# =============================================================================
# env/dag_job.py — DAG Job model
# =============================================================================
from typing import Dict, List
from .task import Task

class DAGJob:
    def __init__(self, job_id: int, app_type: str, arrival_time: float):
        self.job_id: int = job_id
        self.app_type: str = app_type  # vd: 'lightgbm', 'mapreduce'
        self.arrival_time: float = arrival_time
        
        self.tasks: Dict[int, Task] = {} # Lưu trữ các Task theo task_id
        self.edges: Dict[int, List[int]] = {} # Map từ task_id -> list(successor task_id)
        
        self.is_completed: bool = False
        self.finish_time: float = 0.0
        
    def add_task(self, task: Task):
        self.tasks[task.task_id] = task
        
    def add_dependency(self, pred_id: int, succ_id: int):
        """Định nghĩa task_id(succ_id) phải chạy sau task_id(pred_id)"""
        if pred_id not in self.edges:
            self.edges[pred_id] = []
        self.edges[pred_id].append(succ_id)
        
        if pred_id not in self.tasks[succ_id].predecessors:
            self.tasks[succ_id].predecessors.append(pred_id)
        if succ_id not in self.tasks[pred_id].successors:
            self.tasks[pred_id].successors.append(succ_id)
            
    def update_task_completion(self, completed_task_id: int, current_time: float) -> List[Task]:
        """
        Gọi khi một task trong DAG hoàn thành.
        Trả về danh sách các task mới sẵn sàng chạy (ready_to_start = True).
        """
        if self.is_completed:
            return []
            
        completed_task = self.tasks[completed_task_id]
        completed_task.done = True
        
        ready_tasks = []
        
        # Kiểm tra các task đi sau
        for succ_id in completed_task.successors:
            succ_task = self.tasks[succ_id]
            
            # Nếu tất cả các task đi trước của succ_task đều đã done
            all_preds_done = all(self.tasks[p_id].done for p_id in succ_task.predecessors)
            
            if all_preds_done and not succ_task.ready_to_start:
                succ_task.ready_to_start = True
                ready_tasks.append(succ_task)
                
        # Kiểm tra xem toàn bộ Job đã xong chưa
        if all(t.done for t in self.tasks.values()):
            self.is_completed = True
            self.finish_time = current_time
            
        return ready_tasks

    @property
    def latency(self) -> float:
        if not self.is_completed:
            return float("nan")
        return self.finish_time - self.arrival_time
        
    def __repr__(self) -> str:
        status = f"done lat={self.latency*1000:.1f}ms" if self.is_completed else "pending"
        return f"DAGJob(id={self.job_id} type={self.app_type} tasks={len(self.tasks)} {status})"