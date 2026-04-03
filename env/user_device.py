# =============================================================================
# env/user_device.py — User Device
# =============================================================================
from __future__ import annotations
import numpy as np
from typing import List, Optional
from env.node import QueueNode
from env.task import Task
import config as cfg


class UserDevice(QueueNode):
    def __init__(self, user_id: int, rng: np.random.Generator):
        super().__init__(
            node_id=user_id,
            cpu_freq=cfg.USER_CPU_FREQ,
            queue_capacity=0,
            label=f"User-{user_id + 1}",
        )
        self.user_id = user_id
        self._rng    = rng
        self._task_counter = 0
        self._last_arrival_time: Optional[float] = None
        self.offloaded_count: int = 0
        self.local_count:     int = 0

    def generate_tasks(self, dt: float, current_time: float) -> List[Task]:
        lam = cfg.ARRIVAL_RATE * dt
        n   = self._rng.poisson(lam)
        tasks = []
        for _ in range(n):
            task_type = self._sample_type()
            gap = None
            if self._last_arrival_time is not None:
                gap = current_time - self._last_arrival_time
            self._last_arrival_time = current_time
            task = Task(
                task_id           = self._next_id(),
                user_id           = self.user_id,
                task_type         = task_type["name"],
                cycles            = task_type["cycles"],
                input_bits        = task_type["input_bits"],
                arrival_time      = current_time,
                inter_arrival_gap = gap,
            )
            tasks.append(task)
        return tasks

    def enqueue_partial(self, task: Task, current_time: float):
        """
        Đưa local part của partial task vào queue.
        Node sẽ chỉ xử lý task.cycles_local cycles.
        """
        task.queue_start = current_time
        self.queue.append(task)

    def step(self, dt: float, current_time: float) -> List[Task]:
        """
        Override để xử lý đúng cycles cho partial task và set cờ done_local.
        """
        finished = []
        cycles_left = self.cpu_freq * dt

        while cycles_left > 0:
            if self.proc is None:
                if self.queue:
                    self._start_next(current_time)
                else:
                    break

            if self.rem <= cycles_left:
                cycles_left -= self.rem
                self.proc.finish_time = current_time + (self.cpu_freq * dt - cycles_left) / self.cpu_freq
                # [QUAN TRỌNG] Set cờ done_local để môi trường biết local đã xử lý xong phần của nó
                self.proc.done_local  = True 
                finished.append(self.proc)
                self.total_done += 1
                self.proc = None
                self.rem  = 0.0
            else:
                self.rem    -= cycles_left
                cycles_left  = 0

        return finished

    def _start_next(self, current_time: float):
        task = self.queue.popleft()
        task.proc_start = current_time
        self.proc = task
        # Partial task chỉ xử lý cycles_local. Dùng getattr cho an toàn tuyệt đối với DAG tasks.
        is_part = getattr(task, 'is_partial', False) or (0 < getattr(task, 'split_ratio', 0) < 1.0)
        self.rem = task.cycles_local if is_part else task.cycles

    def get_obs(self) -> dict:
        return {
            "user_id":         self.user_id,
            "queue_len":       self.queue_len,
            "is_busy":         self.is_busy,
            "load":            self.load,
            "local_count":     self.local_count,
            "offloaded_count": self.offloaded_count,
        }

    def reset(self):
        super().reset()
        self.offloaded_count    = 0
        self.local_count        = 0
        self._last_arrival_time = None

    def _next_id(self) -> int:
        self._task_counter += 1
        return self.user_id * 10_000_000 + self._task_counter

    def _sample_type(self) -> dict:
        probs = [t["prob"] for t in cfg.TASK_TYPES]
        idx   = self._rng.choice(len(cfg.TASK_TYPES), p=probs)
        return cfg.TASK_TYPES[idx]