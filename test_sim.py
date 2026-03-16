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
    # Kiểm tra bất biến: latency = tx + wait + proc
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

    # Kiểm tra edge_id ghi đúng
    off_rows = [r for r in rows if r["offloaded"] == "1"]
    bad_eid  = [r for r in off_rows if r["edge_id"] not in ("0","1")]
    check(f"edge_id đúng (0 hoặc 1) cho {len(off_rows)} task offloaded",
          len(bad_eid) == 0)

    # In nhanh summary
    data = json.load(open(sj))
    print(f"\n  total_done={data['total_done']}  offload={data['offload_ratio']*100:.1f}%")
    lat = data['latency_all_ms']
    print(f"  latency mean={lat['mean']}ms  p95={lat['p95']}ms  p99={lat['p99']}ms")
    print(f"  inter_arrival_ms có trong CSV: {'inter_arrival_ms' in rows[0]}")
    print(f"  cycles_local/edge có trong CSV: {'cycles_local' in rows[0]}")


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
    print("\n" + "=" * 55)
    print("  DONE")
    print("=" * 55)