import redis
import json
import time
import sys

# 连接Redis
r = redis.Redis(host='thzy-001.redis.rds.aliyuncs.com', port=6379, password="Sonic513", decode_responses=True)
# 构建开始识别消息
start_message = {
    "msg_type": "start",
    "msg_id": "cd5925d0-8983-4445-bc9b-8700da3a5820",
    "ts": int(time.time()),
    "payload": {
        "origin_uri": "/home/user/gx/recon/tests/uav.mp4",
        "labeled_uri": "/home/user/gx/recon/tests/uav_processed.mp4",
        "airport_sn": "xxxxxx",
        "vehicle_sn": "yyyyyy",
        "tenant_id": "aaaaaa",
        "plan_id": "bbbbbb",
        "task_id": "cccccc"
    }
}

# 构建停止识别消息
stop_message = {
    "msg_type": "stop",
    "msg_id": "cd5925d0-8983-4445-bc9b-8700da3a5820",
    "ts": int(time.time()),
    "payload": {
        "origin_uri": "/home/user/gx/recon/tests/uav.mp4",
        "labeled_uri": "/home/user/gx/recon/tests/uav_processed.mp4",
        "airport_sn": "xxxxxx",
        "vehicle_sn": "yyyyyy",
        "tenant_id": "aaaaaa",
        "plan_id": "bbbbbb",
        "task_id": "cccccc"
    }
}

if __name__ == "__main__":
    if len(sys.argv) ==2:
        if sys.argv[1] == "start":
            # 发送消息
            r.publish('cloud_uav:channel:recognize:signal', json.dumps(start_message))
            print("已发送开始识别消息")
        if sys.argv[1] == "stop":
            # 发送消息
            r.publish('cloud_uav:channel:recognize:signal', json.dumps(stop_message))
            print("已发送停止识别消息")
    else:
        print("输入格式为: python redisChannelTester.py start/stop")
