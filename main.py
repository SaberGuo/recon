
from redisChannelListener import VideoRecognitionListener

from recognizer import UAVRecognizer
import json
'''
| 参数         | 类型                | 必含  | 可空  | 示例值                                    | 说明                                          |
| :--------- | ----------------- | --- | --- | -------------------------------------- | ------------------------------------------- |
| `msg_type` | `signal_msg_type` | 是   | 否   | `start_recognize`                      | 消息类型。该参数的类型为枚举，具体内容请参照下方的 `signal_msg_type` |
| `msg_id`   | `str`             | 是   | 否   | `cd5925d0-8983-4445-bc9b-8700da3a5820` | 消息编号                                        |
| `ts`       | `int`             | 是   | 否   | `1757596521`                           | Unix 时间戳（秒级）                                |
| `payload`  | `payload`         | 是   | 否   |                                        | 消息载荷。该参数的类型为结构体，具体内容请参照下方的 `payload`        |
1. `signal_msg_type` 枚举说明

| 值       | 说明   |
| :------ | :--- |
| `start` | 开始识别 |
| `stop`  | 结束识别 |

2. `payload` 结构体说明

| 参数            | 类型    | 必含  | 可空  | 示例值                                  | 说明    |
| :------------ | :---- | :-- | :-- | :----------------------------------- | :---- |
| `origin_uri`  | `str` | 是   | 否   | `rtsp://127.0.0.1:554/live1`         | 拉流地址  |
| `labeled_uri` | `str` | 是   | 否   | `rtsp://127.0.0.1:554/live1/labeled` | 推流地址  |
| `airport_sn`  | `str` | 是   | 否   |                                      | 机场SN  |
| `vehicle_sn`  | `str` | 是   | 否   |                                      | 飞行器SN |
| `tenant_id`   | `str` | 是   | 否   |                                      | 租户编号  |
| `plan_id`     | `str` | 是   | 否   |                                      | 计划ID  |
| `task_id`     | `str` | 是   | 否   |                                      | 任务ID  |
 > 注意：`payload` 的结构可能不尽全，后续业务实际开发过程可能新增字段，但不会减少字段。

示例：
```bash
redis> PUBLISH cloud_uav:channel:recognize:signal {"msg_type": "start", "msg_id": "cd5925d0-8983-4445-bc9b-8700da3a5820", "ts": 1757596521, "payload": {"pull_uri": "rtsp://127.0.0.1:554/live1", "labeled_uri": "rtsp://127.0.0.1:554/live1/labeled", "airport_sn": "xxxxxx", "vehicle_sn": "yyyyyy", "tenant_id": "aaaaaa", "plan_id": "bbbbbb", "task_id": "cccccc"}}
'''

class RecognitionProcessor:
    def __init__(self):
        MODEL_PATHS = [
        "./models/human_car.pt",   # 检测 people 和 car
        "./models/landslide.pt"    # 仅检测 landslide
        ]

        # � 关键：每个模型独立的类别映射（ID → 业务语义名称）
        MODEL_CLASSES = [
            {0: "people", 1: "car"},         # model1: id0=people, id1=car
            {0: "landslide"}                 # model2: id0=landslide
        ]
        self.recognizer = UAVRecognizer(MODEL_PATHS, MODEL_CLASSES)

    

    def process_sign(self, data):
        res = json.loads(data)

        if res["msg_type"] == "start":

        origin_uri = res["payload"]["origin_uri"]
        labeled_uri = res["payload"]["labeled_uri"]
        pass

if __name__ == "__main__":
    # 在后台线程中运行监听器
    rp = RecognitionProcessor()
    listener = VideoRecognitionListener(host="developers:Sonic513@thzy-001.redis.rds.aliyuncs.com", port=6379)
    listener.set_callback(rp.process_sign)

    # 启动后台线程
    import threading
    thread = threading.Thread(target=listener.start_listening, daemon=True)
    thread.start()