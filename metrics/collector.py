# =============================================================================
# metrics/collector.py — Thu thập, lưu trữ và xuất metrics
# =============================================================================
from __future__ import annotations
import os
import json
import csv
from typing import List, Optional
import numpy as np
from env.task import Task
import config as cfg


class MetricsCollector:
    """
    Ghi lại metrics theo từng interval và từng task hoàn thành.

    Xuất:
        - {output_dir}/{run_name}_timeseries.csv  — metrics theo thời gian
        - {output_dir}/{run_name}_tasks.csv        — mỗi task 1 dòng
        - {output_dir}/{run_name}_summary.json     — tổng kết cuối simulation
    """

    def __init__(self, run_name: str, output_dir: str = cfg.METRICS_OUTPUT_DIR):
        self.run_name   = run_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._interval   = cfg.METRICS_INTERVAL
        self._next_snap  = self._interval

        # Buffer task hoàn thành từ lần snapshot trước
        self._buf: List[Task] = []

        # Time-series records
        self.timeseries: List[dict] = []

        # Tất cả task records
        self.task_records: List[dict] = []

        # === Bảng băm lưu phần thưởng và cánh tay của UCB ===
        self.ucb_rewards: dict = {}
        self.ucb_arms: dict = {}

        # === [FIX] Tích lũy reward để vẽ đồ thị convergence ===
        self._cumulative_reward: float = 0.0

    # ── Feed data ─────────────────────────────────────────────────────────────

    def register_ucb_reward(self, task_id: str, reward: float):
        """Lưu lại phần thưởng của task để xuất ra file CSV sau này."""
        self.ucb_rewards[task_id] = reward

    def register_ucb_arm(self, task_id: str, arm_name: str):
        """Lưu lại quyết định (cánh tay) của thuật toán."""
        self.ucb_arms[task_id] = arm_name

    def on_tasks_done(self, tasks: List[Task]):
        """Gọi sau mỗi env.step() với danh sách task vừa hoàn thành."""
        self._buf.extend(tasks)
        for t in tasks:
            self.task_records.append(self._task_row(t))

    def tick(self, sim_time: float, env_obs: dict):
        """
        Gọi sau mỗi env.step().
        Snapshot metrics nếu đã đến interval tiếp theo.
        """
        if sim_time >= self._next_snap:
            snap = self._make_snapshot(sim_time, env_obs)
            self.timeseries.append(snap)
            self._buf.clear()
            self._next_snap += self._interval

    # ── Export ────────────────────────────────────────────────────────────────

    def save_timeseries(self):
        """Xuất CSV time-series."""
        if not self.timeseries:
            return
        path = os.path.join(self.output_dir, f"{self.run_name}_timeseries.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.timeseries[0].keys())
            writer.writeheader()
            writer.writerows(self.timeseries)
        print(f"  [metrics] timeseries → {path}")

    def save_tasks(self):
        """Xuất CSV từng task."""
        if not self.task_records:
            return
        path = os.path.join(self.output_dir, f"{self.run_name}_tasks.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.task_records[0].keys())
            writer.writeheader()
            writer.writerows(self.task_records)
        print(f"  [metrics] tasks      → {path}")

    def save_summary(self, summary: dict):
        """Xuất JSON tổng kết."""
        path = os.path.join(self.output_dir, f"{self.run_name}_summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  [metrics] summary    → {path}")

    def save_all(self, summary: dict):
        self.save_timeseries()
        self.save_tasks()
        self.save_summary(summary)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _make_snapshot(self, sim_time: float, env_obs: dict) -> dict:
        tasks = self._buf
        done_local = [t for t in tasks if not t.offloaded]
        done_edge  = [t for t in tasks if t.offloaded]

        def mean_lat(lst):
            lats = [t.latency * 1000 for t in lst if not np.isnan(t.latency)]
            return round(float(np.mean(lats)), 2) if lats else None

        def p95_lat(lst):
            lats = [t.latency * 1000 for t in lst if not np.isnan(t.latency)]
            return round(float(np.percentile(lats, 95)), 2) if lats else None

        def mean_reward(lst):
            rews = [self.ucb_rewards.get(t.task_id) for t in lst
                    if t.task_id in self.ucb_rewards]
            rews = [r for r in rews if r is not None]
            return round(float(np.mean(rews)), 4) if rews else None

        # === [FIX] Tính cumulative reward cho interval này rồi cộng dồn ===
        interval_rewards = [
            self.ucb_rewards[t.task_id]
            for t in tasks
            if t.task_id in self.ucb_rewards
        ]
        if interval_rewards:
            self._cumulative_reward += float(np.sum(interval_rewards))
        # ==================================================================

        # Queue lengths tổng
        total_user_q = sum(u["queue_len"] for u in env_obs["users"])
        total_edge_q = sum(e["queue_len"] for e in env_obs["edges"])

        row = {
            "sim_time":              round(sim_time, 3),
            "tasks_done_interval":   len(tasks),
            "tasks_local":           len(done_local),
            "tasks_offloaded":       len(done_edge),
            "mean_lat_all_ms":       mean_lat(tasks),
            "mean_lat_local_ms":     mean_lat(done_local),
            "mean_lat_edge_ms":      mean_lat(done_edge),
            "p95_lat_all_ms":        p95_lat(tasks),
            "ucb_reward_mean":       mean_reward(tasks),
            # === [FIX] Cột mới dùng để vẽ đồ thị Convergence ===
            "ucb_reward_cumulative": round(self._cumulative_reward, 4),
            # =====================================================
            "total_user_qlen":       total_user_q,
            "total_edge_qlen":       total_edge_q,
        }

        # Thêm queue length từng node
        for u in env_obs["users"]:
            row[f"u{u['user_id']}_qlen"] = u["queue_len"]
        for e in env_obs["edges"]:
            row[f"e{e['edge_id']}_qlen"]  = e["queue_len"]
            row[f"e{e['edge_id']}_busy"]  = int(e["is_busy"])
            row[f"e{e['edge_id']}_txbuf"] = e.get("tx_buffer", 0)
            row[f"e{e['edge_id']}_load"]  = e.get("total_load", 0)

        return row

    def _task_row(self, t: Task) -> dict:
        return {
            "task_id":           t.task_id,
            "job_id":            getattr(t, 'job_id', -1),
            "app_name":          getattr(t, 'app_type', 'Independent'),
            "node_name":         getattr(t, 'dag_name', 'None'),
            "ucb_reward":        round(self.ucb_rewards.get(t.task_id, 0.0), 4)
                                 if t.task_id in self.ucb_rewards else None,
            "ucb_arm":           self.ucb_arms.get(t.task_id, None)
                                 if t.task_id in self.ucb_arms else None,
            "user_id":           t.user_id,
            "task_type":         t.task_type,
            "offloaded":         int(t.offloaded),
            "edge_id":           t.edge_id if t.offloaded else -1,
            "arrival_time":      round(t.arrival_time,  4),
            "finish_time":       round(t.finish_time,   4),
            "inter_arrival_ms":  round(t.inter_arrival_gap * 1000, 2)
                                 if t.inter_arrival_gap is not None else None,
            "latency_ms":        round(t.latency * 1000, 2)
                                 if not np.isnan(t.latency) else None,
            "waiting_ms":        round(t.waiting_time * 1000, 2)
                                 if not np.isnan(t.waiting_time) else None,
            "proc_ms":           round(t.processing_time * 1000, 2)
                                 if not np.isnan(t.processing_time) else None,
            "tx_delay_ms":       round(t.tx_delay * 1000, 2),
            "transit_ms":        round(t.transit_time * 1000, 2)
                                 if not np.isnan(t.transit_time) else None,
            "channel_rate_mbps": round(t.channel_rate / 1e6, 2)
                                 if t.channel_rate else None,
            "cycles_total":      t.cycles,
            "split_ratio":       round(t.split_ratio, 4),
            "cycles_local":      t.cycles_local,
            "cycles_edge":       t.cycles_edge,
            "local_ratio":       round(t.local_ratio, 4),
            "edge_ratio":        round(t.edge_ratio,  4),
            "finish_time_edge":  round(t.finish_time_edge, 4) if t.offloaded else None,
        }