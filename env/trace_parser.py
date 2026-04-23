# =============================================================================
# env/trace_parser.py — Module đọc dữ liệu trace DAG từ file JSON
# =============================================================================
import json
from typing import Dict
from .task import Task
from .dag_job import DAGJob

class DAGParser:
    def __init__(self):
        self.current_task_id = 0

    def parse_job(self, file_path: str, job_id: int, user_id: int, arrival_time: float, app_type: str) -> DAGJob:
        """
        Đọc file JSON và trả về một đối tượng DAGJob chứa các Task đã được liên kết.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        app_data = data.get("Application", {})
        edges_data = app_data.get("Edges", {})
        vertices_data = app_data.get("Vertices", [])
        
        job = DAGJob(job_id=job_id, app_type=app_type, arrival_time=arrival_time)
        
        # Dictionary dùng để map từ dag_name (ví dụ "0", "s") sang task_id nội bộ
        name_to_id: Dict[str, int] = {}
        
        # 1. Parse các Vertices để tạo danh sách Task
        for v in vertices_data:
            v_name = v["name"]
            
            # Trích xuất dữ liệu dựa theo format của bài báo
            # file: ["name", size_cycles]
            cycles = float(v.get("file", ["NULL", 0])[1])
            
            # model: ["name", size]
            model_size = float(v.get("model", ["NULL", 0])[1])
            
            # Nhân với 8000 để quy đổi từ KB sang Bits (Giả sử file size đang ở đơn vị KB)
            input_bits = (cycles + model_size) * 8000 
        
            # Khởi tạo Task
            task = Task(
                task_id=self.current_task_id,
                user_id=user_id,
                task_type="DAG_Task",
                cycles=cycles,
                input_bits=input_bits,
                arrival_time=arrival_time
            )
            
            # Cập nhật các field dành riêng cho DAG
            task.job_id = job_id
            task.dag_name = v_name
            task.model_size = model_size
            task.app_type = app_type  
            
            name_to_id[v_name] = self.current_task_id
            job.add_task(task)
            self.current_task_id += 1
            
        # 2. Xây dựng Dependency dựa trên Edges
        for pred_name, succ_list in edges_data.items():
            if pred_name not in name_to_id:
                continue
            pred_id = name_to_id[pred_name]
            
            for succ_name in succ_list:
                # Bỏ qua các node kết thúc giả (như "end") nếu nó không có trong Vertices
                if succ_name in name_to_id:
                    succ_id = name_to_id[succ_name]
                    job.add_dependency(pred_id, succ_id)
                    
        # 3. Cập nhật trạng thái ready_to_start
        # Chỉ những task không có predecessor nào (vd node "s") mới được phép chạy ngay
        for task in job.tasks.values():
            task.ready_to_start = (len(task.predecessors) == 0)
            
        return job