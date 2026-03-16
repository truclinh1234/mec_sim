# =============================================================================
# policy/base_policy.py — Abstract Base Class cho mọi offloading policy
# =============================================================================
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from env.task import Task
    from env.mec_env import MecEnv


class BasePolicy(ABC):
    """
    Interface chuẩn cho offloading policy.

    Để thêm thuật toán mới:
        1. Tạo file trong policy/
        2. Kế thừa BasePolicy
        3. Override method decide() (bắt buộc)
        4. Đăng ký trong policy/__init__.py và run_experiment.py

    Method decide() nhận:
        task  : Task đang cần route
        user_obs : dict trạng thái user sinh ra task
        obs   : dict toàn bộ môi trường (users + edges)

    Method decide() trả về:
        'local'         → xử lý tại user device
        int (edge_id)   → offload lên edge server đó
    """

    name: str = "BasePolicy"

    @abstractmethod
    def decide(self, task: "Task", user_obs: dict, obs: dict) -> "str | int":
        """
        Quyết định route cho 1 task.

        Params:
            task:     Task cần route (có .task_type, .cycles, .input_bits)
            user_obs: Trạng thái user tạo ra task
                      Keys: user_id, queue_len, is_busy, load, ...
            obs:      Toàn bộ trạng thái môi trường
                      Keys: sim_time, users[list], edges[list]

        Returns:
            'local'  hoặc  edge_id (int)
        """
        ...

    def act(self, tasks: List["Task"], obs: dict) -> List[Tuple["Task", "str | int"]]:
        """
        Route từng task, cập nhật live_obs sau mỗi quyết định.
        Hỗ trợ 3 loại dest: 'local', int, (int, float).
        """
        live_users = [dict(u) for u in obs["users"]]
        live_edges = [dict(e) for e in obs["edges"]]
        live_obs   = {**obs, "users": live_users, "edges": live_edges}
        actions    = []

        for task in tasks:
            user_obs = live_obs["users"][task.user_id]
            dest     = self.decide(task, user_obs, live_obs)
            actions.append((task, dest))

            # Cập nhật live_obs
            if dest == "local":
                live_obs["users"][task.user_id]["load"]      += 1
                live_obs["users"][task.user_id]["queue_len"] += 1
            elif isinstance(dest, tuple):
                # Partial: cả local lẫn edge đều tăng load
                edge_id = int(dest[0])
                live_obs["users"][task.user_id]["load"]      += 1
                live_obs["edges"][edge_id]["tx_buffer"]  += 1
                live_obs["edges"][edge_id]["total_load"] += 1
            else:
                edge_id = int(dest)
                live_obs["edges"][edge_id]["tx_buffer"]  += 1
                live_obs["edges"][edge_id]["total_load"] += 1

        return actions

    # ── Utility helpers (dùng chung cho các subclass) ────────────────────────

    def _least_loaded_edge(self, obs: dict) -> int:
        """Trả về edge_id của edge ít tải nhất."""
        edges = obs["edges"]
        return min(edges, key=lambda e: e["total_load"])["edge_id"]

    def _most_powerful_edge(self, obs: dict) -> int:
        """Trả về edge_id của edge có CPU cao nhất."""
        edges = obs["edges"]
        return max(edges, key=lambda e: e["cpu_freq"])["edge_id"]

    def _edge_ids(self, obs: dict) -> List[int]:
        return [e["edge_id"] for e in obs["edges"]]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"