# =============================================================================
# policy/ucb_policy.py — Contextual Bandit với Upper Confidence Bound (LinUCB)
# =============================================================================
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Dict, List
import config as cfg
from policy.base_policy import BasePolicy

if TYPE_CHECKING:
    from env.task import Task



class ArmState:
    """Trạng thái LinUCB của 1 arm (Ridge regression online)."""

    def __init__(self, d: int, lam: float = 1.0):
        self.A: np.ndarray = lam * np.eye(d)   # d×d covariance
        self.b: np.ndarray = np.zeros(d)        # d reward accumulator
        self._A_inv: np.ndarray | None = None   # cache

    def update(self, phi: np.ndarray, reward: float):
        self.A += np.outer(phi, phi)
        self.b += reward * phi
        self._A_inv = None                      # invalidate cache

    def A_inv(self) -> np.ndarray:
        if self._A_inv is None:
            self._A_inv = np.linalg.inv(self.A)
        return self._A_inv

    def theta(self) -> np.ndarray:
        return self.A_inv() @ self.b

    def ucb_score(self, phi: np.ndarray, alpha: float) -> float:
        th = self.theta()
        exploit = float(th @ phi)
        explore = alpha * float(np.sqrt(max(phi @ self.A_inv() @ phi, 0.0)))
        return exploit + explore


# ─────────────────────────────────────────────────────────────────────────────

class UCBPolicy(BasePolicy):
    """
    Contextual Bandit (LinUCB) cho MEC offloading.

    Params
    ------
    alpha       : hệ số exploration (lớn → khám phá nhiều hơn)
    deadline_ms : mốc latency dùng tính reward (ms)
    lam         : hệ số regularisation của Ridge (A = λI ban đầu)
    min_pulls   : số lần tối thiểu mỗi arm phải được thử trước khi UCB thuần

    Context vector φ gồm 5 feature (mỗi arm có feature riêng):
        [0] local_load_norm      tải hàng đợi local / 10
        [1] edge_queue_norm      tải hàng đợi edge  / 10
        [2] cpu_freq_norm        tần số CPU edge    / max_freq
        [3] cycles_norm          khối lượng task    / 1e9
        [4] tx_est_norm          ước tính tx delay  / 1.0s
    Với arm 'local', các feature edge được đặt = 0.
    """

    name = "UCB"

    FEATURE_DIM = 5

    def __init__(
        self,
        alpha: float = 1.0,
        deadline_ms: float = 2000.0,    
        lam: float = 1.0,
        min_pulls: int = 20,            
    ):
        self.alpha = alpha
        self.deadline = deadline_ms / 1000.0
        self.lam = lam
        self.min_pulls = min_pulls

        self._arms: Dict[str | int, ArmState] = {}

        self._pull_count: Dict[str | int, int] = {}

        # Buffer: task_id → (arm_key, phi)
        self._pending: Dict[int, tuple] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def decide(self, task: "Task", user_obs: dict, obs: dict) -> "str | int":
        self._init_arms_if_needed(obs)

        phi_map = self._build_context(task, user_obs, obs)

        under_explored = [
            arm for arm in self._arms
            if self._pull_count.get(arm, 0) < self.min_pulls
        ]

        if under_explored:
            # Chọn arm ít được thử nhất (không random hoàn toàn)
            best_arm = min(under_explored, key=lambda a: self._pull_count.get(a, 0))
        else:
            # UCB bình thường sau khi tất cả arm đã được warm-up
            best_arm = max(
                self._arms,
                key=lambda a: self._arms[a].ucb_score(phi_map[a], self.alpha),
            )

        self._pull_count[best_arm] = self._pull_count.get(best_arm, 0) + 1

        # Lưu vào buffer chờ reward
        self._pending[task.task_id] = (best_arm, phi_map[best_arm])

        return best_arm   # 'local' hoặc int edge_id

    def update(self, task: "Task"):
        """
        Gọi sau khi task.done == True để cập nhật model.
        Tích hợp vào main.py: ucb_policy.update(completed_task)
        """
        if task.task_id not in self._pending:
            return
        if not task.done or np.isnan(task.latency):
            self._pending.pop(task.task_id, None)
            return

        arm_key, phi = self._pending.pop(task.task_id)

        reward = max(-1.0, 1.0 - task.latency / self.deadline)

        if arm_key in self._arms:
            self._arms[arm_key].update(phi, reward)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_arms_if_needed(self, obs: dict):
        if self._arms:
            return
        self._arms["local"] = ArmState(self.FEATURE_DIM, self.lam)
        self._pull_count["local"] = 0
        for e in obs["edges"]:
            eid = e["edge_id"]
            self._arms[eid] = ArmState(self.FEATURE_DIM, self.lam)
            self._pull_count[eid] = 0

    def _build_context(
        self,
        task: "Task",
        user_obs: dict,
        obs: dict,
    ) -> Dict["str | int", np.ndarray]:
        """
        Trả về dict arm_key → phi vector (FEATURE_DIM,).
        """
        local_load = user_obs.get("load", 0)
        cycles_norm = task.cycles / 1e9
        
        max_freq = max((e.get("cpu_freq", 1.0) for e in obs["edges"]), default=1.0)
        if max_freq == 0:
            max_freq = 1.0

        local_cpu_norm = cfg.USER_CPU_FREQ / max_freq
        phi_map: Dict[str | int, np.ndarray] = {}

        # Arm local — feature edge = 0
        phi_map["local"] = np.array([
            local_load / 10.0,
            0.0,
            local_cpu_norm,
            cycles_norm,
            0.0,
        ], dtype=float)

        

        DEFAULT_RATE = 20e6

        for e in obs["edges"]:
            eid = e["edge_id"]
            edge_queue = e.get("queue_len", e.get("total_load", 0))
            cpu_freq = e.get("cpu_freq", max_freq)
            rate = e.get("channel_rate", DEFAULT_RATE) or DEFAULT_RATE
            tx_est = task.input_bits / rate

            phi_map[eid] = np.array([
                local_load / 10.0,
                edge_queue / 10.0,
                cpu_freq / max_freq,
                cycles_norm,
                min(tx_est, 1.0),
            ], dtype=float)

        return phi_map

    def __repr__(self) -> str:
        pulls = {k: self._pull_count.get(k, 0) for k in self._arms}
        return (
            f"UCBPolicy(alpha={self.alpha}, "
            f"deadline={self.deadline*1000:.0f}ms, "
            f"min_pulls={self.min_pulls}, "
            f"arms={list(self._arms.keys())}, "
            f"pull_count={pulls})"
        )