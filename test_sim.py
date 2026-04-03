# =============================================================================
# test_sim.py — Kiểm tra mô phỏng: python test_sim.py
# =============================================================================
import csv, json, os, time
import numpy as np
import config as cfg
from env.task        import Task
from env.node        import QueueNode
from env.user_device import UserDevice
from env.edge_server import EdgeServer
from env.mec_env     import MecEnv
from env.dag_job     import DAGJob
from policy.threshold_policy import ThresholdPolicy
from metrics.collector       import MetricsCollector

P = "  ✓"
F = "  ✗ FAIL"

def check(label, ok, got=None, exp=None):
    detail = f"  got={got}  expected={exp}" if got is not None else ""
    print(f"{P if ok else F}  {label}{detail}")
    return ok

def near(a, b, tol=0.02):
    return abs(a - b) / max(abs(b), 1e-9) < tol


# ─── [1] Trace đường đi task ──────────────────────────────────────────────────
def test_trace():
    print("\n[1] TRACE: task đi đâu, vào queue nào, routing lên edge nào")

    cfg.NUM_USERS         = 1
    cfg.EDGE_SERVERS      = [{"id": 0, "cpu_freq": 2e9, "queue_capacity": 0, "label": "Edge-1"}]
    cfg.ARRIVAL_RATE      = 0
    cfg.SIM_DURATION      = 10.0
    cfg.DT                = 0.001
    cfg.CHANNEL_RATE_BPS  = 10e6
    cfg.PROPAGATION_DELAY = 0.0

    env = MecEnv(seed=0); env.reset()

    t_local = Task(1, 0, "Medium", cycles=1e9, input_bits=5e6, arrival_time=0.0)
    t_edge  = Task(2, 0, "Medium", cycles=1e9, input_bits=5e6, arrival_time=0.0)

    env.apply_actions([(t_local, "local"), (t_edge, 0)])

    check("t_local vào queue User-0",        env.users[0].queue_len == 1)
    check("t_edge  vào tx_buffer Edge-0",     len(env.edges[0].tx_buffer) == 1)
    check("t_edge  offloaded=True",          t_edge.offloaded)
    check("t_edge  edge_id=0",               t_edge.edge_id == 0)
    check("t_edge  tx_delay=0.5s",           near(t_edge.tx_delay, 0.5))

    while not env.done and len(env.finished_tasks) < 2:
        env.step()

    check("Cả 2 task done",                  t_local.done and t_edge.done)
    if t_local.done:
        check("Local proc ≈ 1000ms",         near(t_local.processing_time, 1.0),
              f"{t_local.processing_time*1000:.1f}ms", "1000ms")
    if t_edge.done:
        check("Edge tx   ≈ 500ms",           near(t_edge.tx_delay, 0.5),
              f"{t_edge.tx_delay*1000:.1f}ms", "500ms")
        check("Edge proc ≈ 500ms",           near(t_edge.processing_time, 0.5),
              f"{t_edge.processing_time*1000:.1f}ms", "500ms")
        check("Edge lat  ≈ 1000ms",          near(t_edge.latency, 1.0),
              f"{t_edge.latency*1000:.1f}ms", "1000ms")
        check("cycles_edge = cycles",        t_edge.cycles_edge == t_edge.cycles)
        check("cycles_local = 0",            t_edge.cycles_local == 0)


# ─── [2] Timing chính xác ─────────────────────────────────────────────────────
def test_timing():
    print("\n[2] TIMING: proc_time = cycles / cpu_freq")

    cases = [
        ("Light  1GHz", 3e8, 1e9, 0.300),
        ("Medium 1GHz", 1e9, 1e9, 1.000),
        ("Heavy  1GHz", 2e9, 1e9, 2.000),
        ("Medium 3GHz", 1e9, 3e9, 0.333),
        ("Heavy  2GHz", 2e9, 2e9, 1.000),
    ]
    for label, cycles, freq, exp in cases:
        node = QueueNode(0, freq)
        t = Task(1, 0, "X", cycles=cycles, input_bits=0, arrival_time=0.0)
        node.enqueue(t, 0.0)
        sim_t = 0.0
        while not t.done and sim_t < 10:
            node.step(0.001, sim_t); sim_t += 0.001
        check(f"{label} → {exp*1000:.0f}ms",
              t.done and near(t.processing_time, exp),
              f"{t.processing_time*1000:.1f}ms", f"{exp*1000:.0f}ms")


# ─── [3] Queue: 3 task, task 3 phải đợi ──────────────────────────────────────
def test_queue():
    print("\n[3] QUEUE: 3 task inject cùng lúc — task-2 đợi 1s, task-3 đợi 2s")

    node  = QueueNode(0, cpu_freq=1e9)
    tasks = [Task(i+1, 0, "M", cycles=1e9, input_bits=0, arrival_time=0.0)
             for i in range(3)]
    for t in tasks: node.enqueue(t, 0.0)

    check("Trước step: queue_len=3, idle", node.queue_len == 3 and not node.is_busy)
    node.step(0.001, 0.0)
    check("Sau step-1: queue_len=2, busy", node.queue_len == 2 and node.is_busy)

    sim_t, log = 0.0, []
    while sim_t <= 3.1:
        for f in node.step(0.001, sim_t): log.append((f.task_id, sim_t))
        sim_t += 0.001

    if len(log) >= 3:
        check(f"Task-1 xong ≈ 1.0s", near(log[0][1], 1.0), f"{log[0][1]:.3f}s", "1.0s")
        check(f"Task-2 xong ≈ 2.0s", near(log[1][1], 2.0), f"{log[1][1]:.3f}s", "2.0s")
        check(f"Task-3 xong ≈ 3.0s", near(log[2][1], 3.0), f"{log[2][1]:.3f}s", "3.0s")
        check("Task-2 waiting ≈ 1.0s", near(tasks[1].waiting_time, 1.0),
              f"{tasks[1].waiting_time:.3f}s", "1.0s")

    # 2 edge song song
    cfg.PROPAGATION_DELAY = 0.0; cfg.CHANNEL_RATE_BPS = 1e9
    e0 = EdgeServer({"id":0,"cpu_freq":2e9,"queue_capacity":0,"label":"E0"})
    e1 = EdgeServer({"id":1,"cpu_freq":2e9,"queue_capacity":0,"label":"E1"})
    ta = Task(10, 0, "M", 1e9, 1e3, 0.0); ta.tx_delay = 1e3/1e9
    tb = Task(11, 1, "M", 1e9, 1e3, 0.0); tb.tx_delay = 1e3/1e9
    e0.enqueue(ta, 0.0); e1.enqueue(tb, 0.0)
    sim_t = 0.0
    while (not ta.done_edge or not tb.done_edge) and sim_t < 5:
        for t in e0.step(0.001, sim_t): t.done_edge = True
        for t in e1.step(0.001, sim_t): t.done_edge = True
        sim_t += 0.001
    check("2 edge song song: xong cùng lúc",
          ta.done_edge and tb.done_edge and abs(ta.finish_time_edge - tb.finish_time_edge) < 0.001)


# ─── [4] Scale ────────────────────────────────────────────────────────────────
def test_scale():
    print("\n[4] SCALE: wall-time < 5s cho 1000 users / 20 edges")

    cfg.SIM_DURATION = 5.0
    cfg.DT           = 0.01
    cfg.ARRIVAL_RATE = 2.0

    cases = [(6, 2, "đề bài"), (200, 10, "large"), (1000, 20, "stress")]
    for nu, ne, label in cases:
        cfg.NUM_USERS    = nu
        cfg.EDGE_SERVERS = [{"id":i,"cpu_freq":3e9 if i%2==0 else 2e9,
                             "queue_capacity":0,"label":f"E{i}"} for i in range(ne)]
        env = MecEnv(seed=42); env.reset()
        t0  = time.perf_counter()
        while not env.done:
            tasks = env.generate_tasks()
            env.apply_actions(ThresholdPolicy().act(tasks, env.get_obs()))
            env.step()
        wall = time.perf_counter() - t0
        s    = env.summary()
        check(f"{label:8s} ({nu}u/{ne}e): {s['total_done']:5d} done  "
              f"off={s['offload_ratio']*100:.0f}%  wall={wall:.2f}s",
              wall < 5.0)


# ─── [5] Metrics ──────────────────────────────────────────────────────────────
def test_metrics():
    print("\n[5] METRICS: CSV + JSON xuất đúng, latency = tx + wait + proc")

    cfg.NUM_USERS         = 6
    cfg.EDGE_SERVERS      = [{"id":0,"cpu_freq":3e9,"queue_capacity":0,"label":"E1"},
                              {"id":1,"cpu_freq":2e9,"queue_capacity":0,"label":"E2"}]
    cfg.ARRIVAL_RATE      = 2.0
    cfg.SIM_DURATION      = 10.0
    cfg.DT                = 0.01
    cfg.CHANNEL_RATE_BPS  = 10e6
    cfg.PROPAGATION_DELAY = 0.002
    cfg.METRICS_INTERVAL  = 1.0
    cfg.METRICS_OUTPUT_DIR = "results_test"

    env = MecEnv(seed=42); env.reset()
    col = MetricsCollector("test", cfg.METRICS_OUTPUT_DIR)
    prev = 0
    while not env.done:
        tasks = env.generate_tasks()
        env.apply_actions(ThresholdPolicy().act(tasks, env.get_obs()))
        env.step()
        new = env.finished_tasks[prev:]; prev = len(env.finished_tasks)
        col.on_tasks_done(new); col.tick(env.sim_time, env.get_obs())

    summary = env.summary(); summary["policy"] = "Threshold"
    col.save_all(summary)

    ts  = f"{cfg.METRICS_OUTPUT_DIR}/test_timeseries.csv"
    tk  = f"{cfg.METRICS_OUTPUT_DIR}/test_tasks.csv"
    sj  = f"{cfg.METRICS_OUTPUT_DIR}/test_summary.json"
    check("timeseries.csv tồn tại", os.path.exists(ts))
    check("tasks.csv tồn tại",      os.path.exists(tk))
    check("summary.json tồn tại",   os.path.exists(sj))

    rows = list(csv.DictReader(open(tk)))
    errors = 0
    for r in rows[:50]:
        lat     = float(r["latency_ms"])  if r["latency_ms"]  else None
        transit = float(r["transit_ms"])  if r["transit_ms"]  else None
        wait    = float(r["waiting_ms"])  if r["waiting_ms"]  else None
        proc    = float(r["proc_ms"])     if r["proc_ms"]     else None
        if lat and transit is not None and wait is not None and proc:
            if abs(lat - (transit + wait + proc)) > 1.0:
                errors += 1
    check("latency = transit + wait + proc (50 tasks)", errors == 0,
          f"{errors} lỗi", "0")

    off_rows = [r for r in rows if r["offloaded"] == "1"]
    bad_eid  = [r for r in off_rows if r["edge_id"] not in ("0","1")]
    check(f"edge_id đúng (0 hoặc 1) cho {len(off_rows)} task offloaded",
          len(bad_eid) == 0)

    data = json.load(open(sj))
    print(f"\n  total_done={data['total_done']}  offload={data['offload_ratio']*100:.1f}%")
    lat = data['latency_all_ms']
    print(f"  latency mean={lat['mean']}ms  p95={lat['p95']}ms  p99={lat['p99']}ms")
    print(f"  inter_arrival_ms có trong CSV: {'inter_arrival_ms' in rows[0]}")
    print(f"  cycles_local/edge có trong CSV: {'cycles_local' in rows[0]}")


# ─── [6] DAGJob ───────────────────────────────────────────────────────────────
def test_dag_job():
    print("\n[6] DAGJob: load, stage, dispatch, mark_done, advance")

    # ── 6.1  Load 4 app, kiểm tra DAG hợp lệ ────────────────────────────
    print("\n  [6.1] Load 4 app_config.json")
    for path in cfg.DAG_APPS:
        try:
            job = DAGJob(job_id=0, config_path=path, arrival_time=0.0, task_id_start=0)
            check(f"Load OK: {path.split('/')[-2]:12s} "
                  f"nodes={job.num_tasks}  compute={job.num_compute_tasks}  "
                  f"stages={job._num_stages}",
                  job.num_tasks > 0 and job._num_stages > 0)
        except Exception as e:
            check(f"Load {path}", False, got=str(e))

    # ── 6.2  Stage structure: lightgbm ───────────────────────────────────
    print("\n  [6.2] Stage structure — lightgbm")
    job = DAGJob(0, "profile_data/lightgbm/app_config.json",
                 arrival_time=0.0, task_id_start=0)
    print(f"\n{job.stage_summary()}\n")

    # lightgbm: s→0→{1,2}→3→end  =  5 stages
    check("lightgbm: 5 stages",        job._num_stages == 5,
          got=job._num_stages, exp=5)
    check("Stage 0 chứa 's'",          "s"   in job._stages[0])
    check("Stage 1 chứa '0'",          "0"   in job._stages[1])
    check("Stage 2 chứa '1' và '2'",   set(job._stages[2]) == {"1","2"})
    check("Stage 3 chứa '3'",          "3"   in job._stages[3])
    check("Stage 4 chứa 'end'",        "end" in job._stages[4])

    # ── 6.3  Task type mapping ────────────────────────────────────────────
    print("\n  [6.3] Task type mapping")
    #  node '0' : data=10KB  model=0   → total=10  → Light
    #  node '1' : data=15KB  model=1200 → total=1215 → Heavy
    #  node '2' : data=15KB  model=600  → total=615  → Medium
    #  node '3' : data=15KB  model=2200 → total=2215 → Heavy
    check("node '0' → Light",  job._tasks["0"].task_type == "Light",
          got=job._tasks["0"].task_type, exp="Light")
    check("node '1' → Heavy",  job._tasks["1"].task_type == "Heavy",
          got=job._tasks["1"].task_type, exp="Heavy")
    check("node '2' → Medium", job._tasks["2"].task_type == "Medium",  # 615KB
          got=job._tasks["2"].task_type, exp="Medium")
    check("node '3' → Heavy",  job._tasks["3"].task_type == "Heavy",
          got=job._tasks["3"].task_type, exp="Heavy")
    check("node 's'   → Dummy (cycles=0)", job._tasks["s"].cycles == 0)
    check("node 'end' → Dummy (cycles=0)", job._tasks["end"].cycles == 0)

    # ── 6.4  Task fields: job_id, dag_task_id, arrival_time ──────────────
    print("\n  [6.4] Task fields")
    t0 = job._tasks["0"]
    check("job_id gán đúng",        t0.job_id == 0)
    check("dag_task_id gán đúng",   t0.dag_task_id == "0")
    check("arrival_time gán đúng",  t0.arrival_time == 0.0)
    check("task_id_start=0 → tăng dần",
          all(job._tasks[nid].task_id >= 0 for nid in job._tasks))

    # ── 6.5  get_ready_tasks: stage 0 toàn dummy → trả stage 1 ngay ─────
    print("\n  [6.5] get_ready_tasks() lần đầu")
    ready = job.get_ready_tasks()
    check("Lần 1: trả 1 task (node '0', bỏ qua 's')",
          len(ready) == 1, got=len(ready), exp=1)
    check("Task trả về có dag_task_id='0'",
          ready[0].dag_task_id == "0")
    check("current_stage == 1 (đã skip stage 0)",
          job.current_stage == 1, got=job.current_stage, exp=1)

    # ── 6.6  Không dispatch lại task đã dispatch ─────────────────────────
    print("\n  [6.6] Không dispatch lại task đã gửi đi")
    ready2 = job.get_ready_tasks()
    check("Lần 2: trả [] (đã dispatch hết stage 1)",
          ready2 == [], got=len(ready2), exp=0)

    # ── 6.7  mark_done bằng task_id (int) ────────────────────────────────
    print("\n  [6.7] mark_done(task_id: int) → advance stage")
    task_id_of_0 = job._tasks["0"].task_id
    job.mark_done(task_id_of_0)       # dùng int task_id
    check("Sau mark_done('0'): current_stage == 2",
          job.current_stage == 2, got=job.current_stage, exp=2)

    # ── 6.8  Stage 2: song song node '1' và '2' ──────────────────────────
    print("\n  [6.8] Stage 2 — song song")
    ready3 = job.get_ready_tasks()
    dag_ids = {t.dag_task_id for t in ready3}
    check("Stage 2 dispatch {'1','2'} cùng lúc",
          dag_ids == {"1","2"}, got=dag_ids, exp={"1","2"})

    # ── 6.9  mark_done bằng dag_task_id (str) ────────────────────────────
    print("\n  [6.9] mark_done(dag_task_id: str)")
    job.mark_done("1")
    check("Sau mark '1' only: stage vẫn 2 (chờ '2')",
          job.current_stage == 2, got=job.current_stage, exp=2)
    job.mark_done("2")
    check("Sau mark '2': advance sang stage 3",
          job.current_stage == 3, got=job.current_stage, exp=3)

    # ── 6.10 Stage 3: node '3' phải chờ cả '1' lẫn '2' xong ─────────────
    print("\n  [6.10] Stage 3 — node '3' chỉ xuất hiện SAU KHI '1' và '2' done")
    ready4 = job.get_ready_tasks()
    check("Stage 3 trả đúng node '3'",
          len(ready4) == 1 and ready4[0].dag_task_id == "3",
          got=[t.dag_task_id for t in ready4], exp=["3"])

    # ── 6.11 Hoàn thành job ───────────────────────────────────────────────
    print("\n  [6.11] Hoàn thành job")
    job.mark_done("3")
    check("Sau mark '3': is_done=True (stage 4 = 'end' là dummy)",
          job.is_done, got=job.current_stage, exp=f">={job._num_stages}")

    # ── 6.12 task_id_start: không trùng giữa các job ─────────────────────
    print("\n  [6.12] global task_id không trùng giữa 2 job liên tiếp")
    j1 = DAGJob(1, "profile_data/lightgbm/app_config.json",
                task_id_start=0)
    j2 = DAGJob(2, "profile_data/lightgbm/app_config.json",
                task_id_start=j1.num_tasks)
    ids_j1 = {t.task_id for t in j1._tasks.values()}
    ids_j2 = {t.task_id for t in j2._tasks.values()}
    check("task_id không giao nhau giữa job1 và job2",
          ids_j1.isdisjoint(ids_j2),
          got=f"j1={min(ids_j1)}-{max(ids_j1)}  j2={min(ids_j2)}-{max(ids_j2)}")

    # ── 6.13 mark_done với task_id không thuộc job → không crash ─────────
    print("\n  [6.13] mark_done với id lạ → bỏ qua, không crash")
    j3 = DAGJob(3, "profile_data/video_app/app_config.json",
                task_id_start=0)
    try:
        j3.mark_done(99999)     # int không tồn tại
        j3.mark_done("zzz")     # str không tồn tại
        check("mark_done id lạ không raise exception", True)
    except Exception as e:
        check("mark_done id lạ không raise exception", False, got=str(e))

    # ── 6.14 Tất cả 4 app: full dispatch simulation ───────────────────────
    print("\n  [6.14] Full dispatch simulation — 4 apps")
    for path in cfg.DAG_APPS:
        app = path.split("/")[-2]
        job = DAGJob(0, path, task_id_start=0)
        dispatched_types: list[str] = []
        steps = 0
        while not job.is_done and steps < 20:
            batch = job.get_ready_tasks()
            for t in batch:
                dispatched_types.append(t.task_type)
                job.mark_done(t.task_id)
            steps += 1
        check(f"{app:12s}: job done sau {steps} bước, "
              f"dispatched={len(dispatched_types)} compute tasks",
              job.is_done and len(dispatched_types) == job.num_compute_tasks,
              got=len(dispatched_types), exp=job.num_compute_tasks)


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  MEC SIMULATION TEST")
    print("=" * 55)
    test_trace()
    test_timing()
    test_queue()
    test_scale()
    test_metrics()
    test_dag_job()
    print("\n" + "=" * 55)
    print("  DONE")
    print("=" * 55)