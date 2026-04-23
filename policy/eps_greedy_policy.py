# =============================================================================
# policy/eps_greedy_policy.py — Contextual Bandit với Epsilon-Greedy
# =============================================================================
from __future__ import annotations

import random
import numpy as np
from typing import TYPE_CHECKING, Dict
import config as cfg
from policy.base_policy import BasePolicy

if TYPE_CHECKING:
    from env.task import Task


class EpsGreedyArmState:
    """Trạng thái của 1 arm dành cho Epsilon-Greedy."""

    def __init__(self, d: int, lam: float = 1.0):
        self.A: np.ndarray = lam * np.eye(d)
        self.b: np.ndarray = np.zeros(d)
        self._A_inv: np.ndarray | None = None

    def update(self, phi: np.ndarray, reward: float):
        self.A += np.outer(phi, phi)
        self.b += reward * phi
        self._A_inv = None

    def A_inv(self) -> np.ndarray:
        if self._A_inv is None:
            self._A_inv = np.linalg.inv(self.A)
        return self._A_inv

    def theta(self) -> np.ndarray:
        return self.A_inv() @ self.b

    def expected_reward(self, phi: np.ndarray) -> float:
        """Chỉ tính phần thưởng kỳ vọng, không có UCB bonus."""
        th = self.theta()
        return float(th @ phi)


class EpsGreedyPolicy(BasePolicy):
    """
    Contextual Bandit (Epsilon-Greedy) làm Baseline cho MEC offloading.
    """

    name = "EpsGreedy"
    FEATURE_DIM = 5

    def __init__(
        self,
        epsilon: float = 0.1,
        deadline_ms: float = 500.0,
        lam: float = 1.0,
    ):
        self.epsilon = epsilon
        self.deadline = deadline_ms / 1000.0
        self.lam = lam

        self._arms: Dict[str | int, EpsGreedyArmState] = {}
        self._pull_count: Dict[str | int, int] = {}
        self._pending: Dict[int, tuple] = {}

    def decide(self, task: "Task", user_obs: dict, obs: dict) -> "str | int":
        self._init_arms_if_needed(obs)
        phi_map = self._build_context(task, user_obs, obs)

        # Tung đồng xu epsilon
        if random.random() < self.epsilon:
            best_arm = random.choice(list(self._arms.keys()))
        else:
            best_arm = max(
                self._arms,
                key=lambda a: self._arms[a].expected_reward(phi_map[a]),
            )

        self._pull_count[best_arm] = self._pull_count.get(best_arm, 0) + 1
        self._pending[task.task_id] = (best_arm, phi_map[best_arm])
        return best_arm

    def update(self, task: "Task"):
        if task.task_id not in self._pending:
            return
        if not task.done or np.isnan(task.latency):
            self._pending.pop(task.task_id, None)
            return

        arm_key, phi = self._pending.pop(task.task_id)
        reward = max(-1.0, 1.0 - task.latency / self.deadline)

        if arm_key in self._arms:
            self._arms[arm_key].update(phi, reward)

    def _init_arms_if_needed(self, obs: dict):
        if self._arms:
            return
        self._arms["local"] = EpsGreedyArmState(self.FEATURE_DIM, self.lam)
        self._pull_count["local"] = 0
        for e in obs["edges"]:
            eid = e["edge_id"]
            self._arms[eid] = EpsGreedyArmState(self.FEATURE_DIM, self.lam)
            self._pull_count[eid] = 0

    def _build_context(
        self, task: "Task", user_obs: dict, obs: dict,
    ) -> Dict["str | int", np.ndarray]:
        # Khớp 100% với ucb_policy.py của bạn
        local_load = user_obs.get("load", 0)
        cycles_norm = task.cycles / 1e9

        max_freq = max((e.get("cpu_freq", 1.0) for e in obs["edges"]), default=1.0)
        if max_freq == 0:
            max_freq = 1.0

        local_cpu_norm = cfg.USER_CPU_FREQ / max_freq
        phi_map: Dict[str | int, np.ndarray] = {}

        phi_map["local"] = np.array([local_load / 10.0, 0.0, local_cpu_norm, cycles_norm, 0.0], dtype=float)

        
        DEFAULT_RATE = 20e6

        for e in obs["edges"]:
            eid = e["edge_id"]
            edge_queue = e.get("queue_len", e.get("total_load", 0))
            cpu_freq = e.get("cpu_freq", max_freq)
            rate = e.get("channel_rate", DEFAULT_RATE) or DEFAULT_RATE
            tx_est = task.input_bits / rate

            phi_map[eid] = np.array([
                local_load / 10.0, edge_queue / 10.0, cpu_freq / max_freq, cycles_norm, min(tx_est, 1.0)
            ], dtype=float)

        return phi_map

    def __repr__(self) -> str:
        return f"EpsGreedyPolicy(eps={self.epsilon}, deadline={self.deadline*1000:.0f}ms)"