# =============================================================================
# main.py — Chạy simulation MEC với UCBPolicy  (ĐÃ FIX)
# Kết quả lưu vào results/  →  sau đó chạy plot.py để xem đồ thị
# Sử dụng: python main.py
# =============================================================================
import os, random, math
import numpy as np
import config as cfg
from env.mec_env import MecEnv
from controller import Controller
from policy.ucb_policy import UCBPolicy
from metrics.collector import MetricsCollector
from env.trace_parser import DAGParser


# ── Cấu hình chạy ─────────────────────────────────────────────────────────────
ALPHA       = 1.0     # UCB exploration coefficient
DEADLINE_MS = 500.0   # ngưỡng latency tính reward (ms)
# ─────────────────────────────────────────────────────────────────────────────


def main():
    os.makedirs("results", exist_ok=True)

    policy = UCBPolicy(alpha=ALPHA, deadline_ms=DEADLINE_MS)
    env    = MecEnv(seed=cfg.RANDOM_SEED)
    ctrl   = Controller(policy=policy)
    col    = MetricsCollector(run_name="sim")
    parser = DAGParser()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    available_apps = [
        ("lightgbm",  os.path.join(base_dir, "profile_data", "lightgbm.json")),
        ("mapreduce", os.path.join(base_dir, "profile_data", "mapreduce.json")),
        ("matrix",    os.path.join(base_dir, "profile_data", "matrix_app.json")),
        ("video",     os.path.join(base_dir, "profile_data", "video_app.json")),
    ]

    active_jobs           = {}
    dag_tasks_to_schedule = []
    prev                  = 0
    next_job_id           = 1
    env.reset()

    print(f"--- BẮT ĐẦU  policy={repr(policy)}  duration={cfg.SIM_DURATION}s ---")

    while not env.done:
        tasks = []

        num_arrivals = np.random.poisson(cfg.ARRIVAL_RATE * cfg.DT)
        for _ in range(num_arrivals):
            app_name, app_path = random.choice(available_apps)
            u_id = random.randint(0, cfg.NUM_USERS - 1)
            new_job = parser.parse_job(
                file_path=app_path, job_id=next_job_id,
                user_id=u_id, arrival_time=env.sim_time, app_type=app_name,
            )
            active_jobs[new_job.job_id] = new_job
            next_job_id += 1
            dag_tasks_to_schedule.extend(
                t for t in new_job.tasks.values() if t.ready_to_start
            )

        if dag_tasks_to_schedule:
            tasks.extend(dag_tasks_to_schedule)
            dag_tasks_to_schedule = []

        ctrl.step(env, tasks)
        env.step()

        new_done = env.finished_tasks[prev:]
        prev = len(env.finished_tasks)

        for task in new_done:
            # FIX: Lấy arm_key TRƯỚC khi policy.update() gọi .pop()
            # Thứ tự đúng:
            #   1. Đọc arm_key từ _pending  (còn trong dict)
            #   2. Gọi policy.update()       (sẽ .pop() task khỏi _pending)
            #   3. Đăng ký vào collector
            arm_key = None
            if hasattr(policy, '_pending') and task.task_id in policy._pending:
                # _pending[task_id] = (arm_key, phi)  →  lấy index [0]
                arm_key = policy._pending[task.task_id][0]

            # Cập nhật UCB model (sẽ pop task khỏi _pending)
            policy.update(task)

            # Tính reward và lưu vào collector
            if task.done and not math.isnan(task.latency):
                reward = max(-1.0, 1.0 - task.latency / policy.deadline)
                col.register_ucb_reward(task.task_id, reward)

                if arm_key is not None:
                    display_arm = "Local" if arm_key == "local" else f"Edge {arm_key}"
                    col.register_ucb_arm(task.task_id, display_arm)

            # Cập nhật DAG: mở khoá task tiếp theo
            if getattr(task, "job_id", None) is not None:
                job = active_jobs.get(task.job_id)
                if job and not job.is_completed:
                    dag_tasks_to_schedule.extend(
                        job.update_task_completion(task.task_id, env.sim_time)
                    )

        col.on_tasks_done(new_done)
        col.tick(env.sim_time, env.get_obs())

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    summary = env.summary()
    summary["policy"] = repr(policy)

    completed_dags = [j for j in active_jobs.values() if j.is_completed]
    summary["dag_completed"]  = len(completed_dags)
    summary["dag_total"]      = len(active_jobs)
    summary["dag_latency_ms"] = (
        sum(j.latency for j in completed_dags) / len(completed_dags) * 1000
        if completed_dags else 0.0
    )

    col.save_all(summary)

    lat = summary.get("latency_all_ms", {})
    print(f"\n--- KẾT QUẢ ---")
    print(f"Tasks done : {summary['total_done']}")
    print(f"Offload%   : {summary['offload_ratio']*100:.1f}%")
    print(f"Lat mean   : {lat.get('mean','—')} ms")
    print(f"Lat p95    : {lat.get('p95','—')} ms")
    print(f"DAG done   : {len(completed_dags)} / {len(active_jobs)}")
    print(f"\n→ Chạy 'python plot.py' để xem đồ thị")


if __name__ == "__main__":
    main()