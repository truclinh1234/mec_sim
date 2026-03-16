# =============================================================================
# policy/threshold_policy.py
# =============================================================================
from __future__ import annotations
from policy.base_policy import BasePolicy
import config as cfg


class ThresholdPolicy(BasePolicy):
    """
    Quyết định routing + split ratio động theo trạng thái hệ thống.

    Logic:
        1. Nếu local rảnh (load < threshold) → full local
        2. Nếu local đầy → tính split ratio động:
               ratio = local_load / (local_load + edge_load + 1)
           Nghĩa là: local càng tắc, edge càng rảnh → ratio càng cao
        3. Nếu Heavy task → luôn offload (ratio từ config hoặc động)

    Params:
        threshold            : ngưỡng load local để bắt đầu offload
        heavy_always_offload : Heavy luôn offload
        use_partial          : True = tính ratio động, False = full offload
    """

    name = "Threshold"

    def __init__(
        self,
        threshold:            int  = cfg.THRESHOLD_QUEUE_LEN,
        heavy_always_offload: bool = cfg.THRESHOLD_HEAVY_ALWAYS,
        use_partial:          bool = False,
    ):
        self.threshold            = threshold
        self.heavy_always_offload = heavy_always_offload
        self.use_partial          = use_partial

    def decide(self, task, user_obs: dict, obs: dict):
        local_load = user_obs["load"]
        should_offload = (
            (self.heavy_always_offload and task.task_type == "Heavy")
            or local_load >= self.threshold
        )

        if not should_offload:
            return "local"

        edge_id   = self._least_loaded_edge(obs)
        edge_load = obs["edges"][edge_id]["total_load"]

        if not self.use_partial:
            return edge_id   # full offload

        # ── Tính split ratio động ─────────────────────────────────────────
        # local_load cao + edge_load thấp → ratio cao (đẩy nhiều lên edge)
        # local_load thấp + edge_load cao → ratio thấp (giữ nhiều ở local)
        #
        # ratio = local_load / (local_load + edge_load + 1)
        # clamp vào [0.1, 0.9] để tránh degenerate
        ratio = local_load / (local_load + edge_load + 1)
        ratio = max(0.1, min(0.9, ratio))

        return (edge_id, ratio)

    def __repr__(self) -> str:
        mode = "partial_dynamic" if self.use_partial else "full_offload"
        return (f"ThresholdPolicy(thr={self.threshold},"
                f" heavy_off={self.heavy_always_offload},"
                f" mode={mode})")