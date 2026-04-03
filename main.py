# =============================================================================
# main.py — Chạy simulation: python main.py
# =============================================================================
import os
import random
import config as cfg
from env.mec_env import MecEnv
from controller import Controller
from policy.threshold_policy import ThresholdPolicy
from metrics.collector import MetricsCollector
from env.trace_parser import DAGParser

def run():
    env  = MecEnv(seed=cfg.RANDOM_SEED)
    ctrl = Controller(policy=ThresholdPolicy(threshold=2, use_partial=True))
    col  = MetricsCollector(run_name="sim")
    parser = DAGParser() 

    # Cấu trúc lưu trữ các DAG đang chạy
    active_jobs = {}
    dag_tasks_to_schedule = []

    # Định nghĩa danh sách các ứng dụng từ file JSON đã tạo ở Bước 1
    base_dir = os.path.dirname(os.path.abspath(__file__))
    available_apps = [
        ("lightgbm", os.path.join(base_dir, "profile_data", "lightgbm.json")),
        ("mapreduce", os.path.join(base_dir, "profile_data", "mapreduce.json")),
        ("matrix", os.path.join(base_dir, "profile_data", "matrix_app.json")),
        ("video", os.path.join(base_dir, "profile_data", "video_app.json"))
    ]

    prev = 0
    next_job_id = 1
    env.reset()

    print(f"--- BẮT ĐẦU MÔ PHỎNG (Thời gian: {cfg.SIM_DURATION}s) ---")

    while not env.done:
        # 1. Sinh task độc lập từ logic môi trường cũ (có thể giữ hoặc comment lại nếu chỉ muốn test DAG)
        # tasks = env.generate_tasks()
        tasks = []
        
        # =====================================================================
        # 2. LOGIC SINH DAG JOB ĐỘNG (Dựa trên Arrival Rate)
        # =====================================================================
        # Xác suất có 1 ứng dụng mới đến trong 1 khoảng DT
        prob_arrival = cfg.ARRIVAL_RATE * cfg.DT
        if random.random() < prob_arrival:
            # Chọn ngẫu nhiên 1 ứng dụng và 1 User hợp lệ
            app_name, app_path = random.choice(available_apps)
            u_id = random.randint(0, cfg.NUM_USERS - 1)
            
            # Khởi tạo Job mới
            new_job = parser.parse_job(
                file_path=app_path, 
                job_id=next_job_id, 
                user_id=u_id, 
                arrival_time=env.sim_time, 
                app_type=app_name
            )
            active_jobs[new_job.job_id] = new_job
            next_job_id += 1
            
            # Lấy các node khởi đầu (VD: node 's') đưa vào hàng đợi
            dag_tasks_to_schedule.extend([t for t in new_job.tasks.values() if t.ready_to_start])
        # =====================================================================

        # 3. Gộp các task DAG đã được mở khóa vào danh sách cần phân bổ
        if dag_tasks_to_schedule:
            tasks.extend(dag_tasks_to_schedule)
            dag_tasks_to_schedule = []  # Đã giao cho controller thì xóa đi

        # 4. Controller ra quyết định offloading
        ctrl.step(env, tasks)       
        
        # 5. Môi trường chạy 1 step thời gian
        env.step()

        # 6. Lấy các task vừa hoàn thành
        new_done = env.finished_tasks[prev:]
        prev = len(env.finished_tasks)
        
        # 7. KÍCH HOẠT DEPENDENCY: Cập nhật DAG và lấy task đi sau
        for completed_task in new_done:
            # Kiểm tra xem task này có thuộc DAG nào không
            if getattr(completed_task, 'job_id', None) is not None: 
                job = active_jobs.get(completed_task.job_id)
                if job and not job.is_completed:
                    # Đánh dấu xong, trả về list các task tiếp theo đã đủ điều kiện chạy
                    next_tasks = job.update_task_completion(completed_task.task_id, env.sim_time)
                    dag_tasks_to_schedule.extend(next_tasks)

        col.on_tasks_done(new_done)
        col.tick(env.sim_time, env.get_obs())

    # --- KẾT THÚC MÔ PHỎNG VÀ IN KẾT QUẢ ---
    summary = env.summary()
    summary["policy"] = repr(ctrl.policy)
    col.save_all(summary)

    # Thống kê hiệu suất của các DAG Job
    completed_dags = [j for j in active_jobs.values() if j.is_completed]
    avg_dag_latency = sum(j.latency for j in completed_dags) / len(completed_dags) if completed_dags else 0.0

    s = summary
    print(f"\nDone      : {s['total_done']} / {s.get('total_generated', '?')}")
    print(f"Offload%  : {s['offload_ratio']*100:.1f}%")
    print(f"DAG Jobs  : Hoàn thành {len(completed_dags)} / Tổng sinh {len(active_jobs)} ứng dụng.")
    print(f"DAG Latency: {avg_dag_latency*1000:.1f}ms (Trung bình)")
    
    lat = s.get('latency_all_ms', {'mean': 0, 'p95': 0})
    print(f"Task Latency: mean={lat['mean']}ms  p95={lat['p95']}ms")


if __name__ == "__main__":
    run()