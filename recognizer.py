'''
file_name: recognizer.py
description: 使用两个YOLOv8模型进行无人机视频帧的目标识别，输出标准化结果。
time: 2025-9-12
version: 1.0
'''

import json
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any, Optional
import time

class UAVRecognizer:
    def __init__(self, model_paths: List[str], model_classes: List[Dict[int, str]]):
        """
        :param model_paths: 两个 YOLOv8 .pt 模型路径列表，如 ["model1.pt", "model2.pt"]
        :param model_classes: 每个模型对应的类别映射字典列表。
                             如: [
                                 {0: "people", 1: "car"},      # model1 的类别
                                 {0: "landslide"}               # model2 的类别
                             ]
        """
        self.models = [YOLO(path) for path in model_paths]
        self.model_classes = model_classes  # 每个模型独立的类别映射
        print(f"[UAVRecognizer] 加载 {len(self.models)} 个模型完成")
        for i, cls_map in enumerate(model_classes):
            print(f"  Model {i+1}: {cls_map}")

    def recognize_frame(
        self,
        frame: np.ndarray,
        lat: float,
        lon: float,
        alt: Optional[float],
        vehicle_info: Dict,
        msg_id: str,
        pull_uri: str,
        labeled_uri: str,
        airport_sn: str,
        vehicle_sn: str,
        tenant_id: str,
        plan_id: str,
        task_id: str,
        frame_ts_ms: int = None
    ) -> Dict[str, Any]:
        """
        对单帧进行双模型识别，输出标准结果格式

        :param frame: OpenCV 读取的 BGR 帧 (numpy array)
        :param lat: 固定纬度
        :param lon: 固定经度
        :param alt: 固定海拔（可为 None）
        :param vehicle_info: 飞行器状态信息（原样保留）
        :param msg_id, pull_uri, ...: 来自平台的原始 payload 字段（用于回填）
        :param frame_ts_ms: 当前帧时间戳（毫秒），若未提供则用当前时间
        :return: 标准 result JSON 字典
        """

        if frame_ts_ms is None:
            frame_ts_ms = int(time.time() * 1000)

        all_detections = []

        # 遍历每个模型及其对应的类别映射
        for model_idx, model in enumerate(self.models):
            classes_map = self.model_classes[model_idx]  # 当前模型的类别映射

            results = model.track(
                frame,
                persist=True,
                conf=0.3,
                iou=0.5,
                tracker="bytetrack.yaml",
                verbose=False
            )[0]

            if results.boxes is not None:
                for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    conf = float(results.boxes.conf.cpu().numpy())

                    # 使用该模型专属的类别映射获取名称
                    class_name = classes_map.get(int(cls_id))
                    if class_name is None:
                        continue  # 跳过未知类别（安全处理）

                    detection = {
                        "target_class": class_name,  # 👈 严格使用模型自身的类别名
                        "frame_ts": frame_ts_ms,
                        "box": {
                            "x_1": x1,
                            "y_1": y1,
                            "x_2": x2,
                            "y_2": y2
                        },
                        "position": {
                            "latitude": lat,
                            "longitude": lon,
                            "altitude": alt
                        },
                        "conf": conf,
                        "track_id": None,  # 👈 禁用 track_id，避免误导
                        "vehicle_info": vehicle_info
                    }
                    all_detections.append(detection)

        # 构造最终结果
        result = {
            "video_path": "",  # 不保存视频片段
            "ts": int(time.time()),  # 发送时的时间戳（秒级）
            "meta_info": {
                "msg_id": msg_id,
                "pull_uri": pull_uri,
                "labeled_uri": labeled_uri,
                "airport_sn": airport_sn,
                "vehicle_sn": vehicle_sn,
                "tenant_id": tenant_id,
                "plan_id": plan_id,
                "task_id": task_id
            },
            "results": all_detections
        }

        return result


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import cv2

    # 👇 请根据你的实际模型路径和类别修改
    MODEL_PATHS = [
        "./weights/human_car.pt",   # 检测 people 和 car
        "./weights/landslide.pt"    # 仅检测 landslide
    ]

    # 👇 关键：每个模型独立的类别映射（ID → 业务语义名称）
    MODEL_CLASSES = [
        {0: "people", 1: "car"},         # model1: id0=people, id1=car
        {0: "landslide"}                 # model2: id0=landslide
    ]

    # 创建识别器
    recognizer = UAVRecognizer(MODEL_PATHS, MODEL_CLASSES)

    # 模拟输入帧（替换为你的视频流帧）
    frame = cv2.imread(r"D:\Doctor1\项目\照片\DJI_20250821130219_0056_V.JPG")
    if frame is None:
        raise FileNotFoundError("请提供测试帧 test_frame.jpg")

    # 固定经纬度（你提供的）
    LAT = 25.070270
    LON = 102.684488
    ALT = 1920.12

    # 飞行器信息（从 Redis 或其他来源获取）
    VEHICLE_INFO = {
        "vehicle": {
            "roll": 30,
            "pitch": 45,
            "yaw": 60,
            "latitude": LAT,
            "longitude": LON,
            "altitude": ALT
        },
        "gimbal": {
            "roll": 30,
            "pitch": 45,
            "yaw": 60
        }
    }

    # 平台原始消息字段
    MSG_ID = "cd5925d0-8983-4445-bc9b-8700da3a5820"
    PULL_URI = "rtsp://127.0.0.1:554/live1"
    LABELED_URI = "rtsp://127.0.0.1:554/live1/labeled"
    AIRPORT_SN = "xxxxxx"
    VEHICLE_SN = "yyyyyy"
    TENANT_ID = "aaaaaa"
    PLAN_ID = "bbbbbb"
    TASK_ID = "cccccc"

    # 执行识别
    result = recognizer.recognize_frame(
        frame=frame,
        lat=LAT,
        lon=LON,
        alt=ALT,
        vehicle_info=VEHICLE_INFO,
        msg_id=MSG_ID,
        pull_uri=PULL_URI,
        labeled_uri=LABELED_URI,
        airport_sn=AIRPORT_SN,
        vehicle_sn=VEHICLE_SN,
        tenant_id=TENANT_ID,
        plan_id=PLAN_ID,
        task_id=TASK_ID,
        frame_ts_ms=int(time.time() * 1000)
    )

    # 输出结果（可直接 publish 到 Redis）
    print(json.dumps(result, indent=2, ensure_ascii=False))