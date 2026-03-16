# =============================================================================
# main.py — Chạy simulation: python main.py
# =============================================================================
import config as cfg
from env.mec_env import MecEnv
from controller import Controller
from policy.threshold_policy import ThresholdPolicy
from metrics.collector import MetricsCollector


def run():
    env  = MecEnv(seed=cfg.RANDOM_SEED)
    ctrl = Controller(policy=ThresholdPolicy(threshold=2, use_partial=True))
    col  = MetricsCollector(run_name="sim")
    prev = 0

    env.reset()
    while not env.done:
        tasks = env.generate_tasks()
        ctrl.step(env, tasks)       # Observer → Policy → Action
        env.step()

        new_done = env.finished_tasks[prev:]
        prev = len(env.finished_tasks)
        col.on_tasks_done(new_done)
        col.tick(env.sim_time, env.get_obs())

    summary = env.summary()
    summary["policy"] = repr(ctrl.policy)
    col.save_all(summary)

    s = summary
    print(f"Done      : {s['total_done']} / {s['total_generated']}")
    print(f"Offload%  : {s['offload_ratio']*100:.1f}%")
    lat = s['latency_all_ms']
    print(f"Latency   : mean={lat['mean']}ms  p95={lat['p95']}ms")
    print(f"By type:")
    for t, st in s['by_type'].items():
        print(f"  {t:8s} count={st['count']:4d}  mean={st['mean_ms']}ms  offload={st['offload_pct']}%")


if __name__ == "__main__":
    run()