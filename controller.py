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
    def observe(self, env: MecEnv) -> dict:
        return env.get_obs()


class Action:
    def execute(self, env: MecEnv, actions: List[Tuple[Task, "str | int"]]):
        env.apply_actions(actions)


class Controller:

    def __init__(self, policy: BasePolicy):
        self.observer = Observer()
        self.policy   = policy
        self.action   = Action()

    def step(self, env: MecEnv, tasks: List[Task]):
        obs     = self.observer.observe(env)       # 1. Quan sát toàn cục
        actions = self._act_with_user_obs(tasks, obs, env)  # 2. Quyết định
        self.action.execute(env, actions)           # 3. Thực thi

    def _act_with_user_obs(
        self,
        tasks: List[Task],
        obs: dict,
        env: MecEnv,
    ) -> List[Tuple[Task, "str | int"]]:
        if not tasks:
            return []

        # UCBPolicy (và bất kỳ policy nào có method decide)
        if hasattr(self.policy, 'decide'):
            actions = []
            # Build lookup user_obs theo user_id
            user_obs_map = {u["user_id"]: u for u in obs["users"]}
            for task in tasks:
                user_obs = user_obs_map.get(task.user_id, {})
                dest = self.policy.decide(task, user_obs, obs)
                actions.append((task, dest))
            return actions

        # Fallback: base_policy.act() cho ThresholdPolicy / RandomPolicy
        return self.policy.act(tasks, obs)