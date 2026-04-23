# =============================================================================
# main.py — Chạy simulation MEC 
# Kết quả lưu vào results/  →  sau đó chạy plot.py để xem đồ thị
# Sử dụng: python main.py
# =============================================================================

import os, random, math
import numpy as np
import config as cfg
from env.mec_env import MecEnv
from controller import Controller
from policy.ucb_policy import UCBPolicy
from policy.eps_greedy_policy import EpsGreedyPolicy
from metrics.collector import MetricsCollector
from env.trace_parser import DAGParser

# ── Cấu hình chạy ─────────────────────────────────────────────────────────────
ALPHA       = 1.0     # UCB exploration coefficient
EPSILON     = 0.1     # Eps-Greedy exploration prob
DEADLINE_MS = 500.0   # ngưỡng latency tính reward (ms)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    available_apps = [
        ("lightgbm",  os.path.join(base_dir, "profile_data", "lightgbm.json")),
        ("mapreduce", os.path.join(base_dir, "profile_data", "mapreduce.json")),
        ("matrix",    os.path.join(base_dir, "profile_data", "matrix_app.json")),
        ("video",     os.path.join(base_dir, "profile_data", "video_app.json")),
    ]

    # Danh sách các policy muốn chạy so sánh
    policies_to_test = [
        # Nhóm LinUCB với các mức độ "tò mò" khác nhau
        ("UCB_alpha_0.1", UCBPolicy(alpha=0.1, deadline_ms=DEADLINE_MS)), # Ít khám phá, chủ yếu khai thác
        ("UCB_alpha_1.0", UCBPolicy(alpha=1.0, deadline_ms=DEADLINE_MS)), # Cân bằng 
        ("UCB_alpha_2.5", UCBPolicy(alpha=2.5, deadline_ms=DEADLINE_MS)), # Cực kỳ tò mò, hay thử Edge mới
        
        # Nhóm Baseline với các tỷ lệ ngẫu nhiên khác nhau
        ("Eps_0.05", EpsGreedyPolicy(epsilon=0.05, deadline_ms=DEADLINE_MS)), # 5% chọn bừa
        ("Eps_0.20", EpsGreedyPolicy(epsilon=0.20, deadline_ms=DEADLINE_MS))  # 20% chọn bừa (rất hỗn loạn)
    ]

    for run_name, policy in policies_to_test:
        print(f"\n==================================================================")
        print(f" BẮT ĐẦU CHẠY MÔ PHỎNG: {run_name} | duration={cfg.SIM_DURATION}s")
        print(f"==================================================================")

        # 1. Khởi tạo lại môi trường cho mỗi policy
        # Đặt lại seed ĐỂ ĐẢM BẢO 2 policy gặp cùng 1 lượng task y hệt nhau
        random.seed(cfg.RANDOM_SEED)
        np.random.seed(cfg.RANDOM_SEED)
        
        env    = MecEnv(seed=cfg.RANDOM_SEED)
        ctrl   = Controller(policy=policy)
        col    = MetricsCollector(run_name=run_name) # File kết quả sẽ có tên là UCB.csv và EpsGreedy.csv
        parser = DAGParser()

        active_jobs           = {}
        dag_tasks_to_schedule = []
        prev                  = 0
        next_job_id           = 1
        env.reset()

        # 2. Vòng lặp mô phỏng
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
                arm_key = None
                if hasattr(policy, '_pending') and task.task_id in policy._pending:
                    arm_key = policy._pending[task.task_id][0]

                policy.update(task)

                if task.done and not math.isnan(task.latency):
                    reward = max(-1.0, 1.0 - task.latency / policy.deadline)
                    col.register_ucb_reward(task.task_id, reward)

                    if arm_key is not None:
                        display_arm = "Local" if arm_key == "local" else f"Edge {arm_key}"
                        col.register_ucb_arm(task.task_id, display_arm)

                if getattr(task, "job_id", None) is not None:
                    job = active_jobs.get(task.job_id)
                    if job and not job.is_completed:
                        dag_tasks_to_schedule.extend(
                            job.update_task_completion(task.task_id, env.sim_time)
                        )

            col.on_tasks_done(new_done)
            col.tick(env.sim_time, env.get_obs())

        # 3. Tổng kết cho Policy hiện tại
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
        print(f"--- KẾT QUẢ CỦA {run_name} ---")
        print(f"Tasks done : {summary['total_done']}")
        print(f"Offload%   : {summary['offload_ratio']*100:.1f}%")
        print(f"Lat mean   : {lat.get('mean','—')} ms")
        print(f"Lat p95    : {lat.get('p95','—')} ms")
        print(f"DAG done   : {len(completed_dags)} / {len(active_jobs)}")

    print(f"\n→ Chạy 'python plot.py' để xem đồ thị so sánh!")

if __name__ == "__main__":
    main()