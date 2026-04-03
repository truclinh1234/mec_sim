# =============================================================================
# controller.py
#
# Controller
# ├── Observer  — đọc trạng thái từ env
# ├── Policy    — ra quyết định
# └── Action    — thực thi vào env
# =============================================================================
import config as cfg
from env.mec_env import MecEnv
from env.task import Task
from policy.base_policy import BasePolicy
from typing import List, Tuple


class Observer:
    """Đọc trạng thái môi trường, trả về obs dict cho Policy."""

    def observe(self, env: MecEnv) -> dict:
        return env.get_obs()


class Action:
    """Nhận quyết định từ Policy, thực thi vào env."""

    def execute(self, env: MecEnv, actions: List[Tuple[Task, "str | int"]]):
        env.apply_actions(actions)


class Controller:
    """
    Điều phối Observer → Policy → Action mỗi time step.

    Dùng trong main.py:
        ctrl = Controller(policy=ThresholdPolicy())
        while not env.done:
            tasks = env.generate_tasks()
            ctrl.step(env, tasks)
            env.step()
    """

    def __init__(self, policy: BasePolicy):
        self.observer = Observer()
        self.policy   = policy
        self.action   = Action()

    def step(self, env: MecEnv, tasks: List[Task]):
        obs     = self.observer.observe(env)       # 1. Quan sát
        actions = self.policy.act(tasks, obs)      # 2. Quyết định
        self.action.execute(env, actions)           # 3. Thực thi