import redis
import signal
import json
import threading
from threading import Thread
import logging
import time
import uuid
from typing import Dict, Any, Optional, List
import cv2
import os
import queue
from datetime import datetime
import sys

from recognizer import UAVRecognizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VideoRecognitionService')

class VideoRecognitionService:
    def __init__(self, host='localhost', port=6379, password=None, db=0):
        """
        初始化视频识别服务
        
        Args:
            host: Redis主机地址
            port: Redis端口
            password: Redis密码
            db: Redis数据库编号
        """
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )
        
        # 频道名称
        self.signal_channel = 'cloud_uav:channel:recognize:signal'
        self.callback_channel = 'cloud_uav:channel:recognize:callback'
        self.result_channel = 'cloud_uav:channel:recognize:result'
        
        # 视频识别线程控制
        self.recognition_thread: Optional[threading.Thread] = None
        self.is_recognition_running = False
        self.should_stop = False

        # � 请根据你的实际模型路径和类别修改
        MODEL_PATHS = [
            "./models/visdrone.pt",   # 检测 people 和 car
            "./models/landslide.pt"    # 仅检测 landslide
        ]

        # � 关键：每个模型独立的类别映射（ID → 业务语义名称）
        MODEL_CLASSES = [
            {0: "pedestrian", 1: "person", 2: "bicyle", 3: "car", 4: "van",
            5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"},
            {0: "landslide"}
        ]
        self.recon = UAVRecognizer(MODEL_PATHS, MODEL_CLASSES)
        if self.recon is None:
            logger.warning("recongnizer初始化失败")
        # 当前任务信息
        self.current_task: Optional[Dict[str, Any]] = None
        

        #视频记录
        thread_stop_event = threading.Event()
        
        self.short_video_queue = queue.Queue()
        short_video_thread = Thread(
            target=self.write_short_video, args=(self.short_video_queue, thread_stop_event)
        )
        short_video_thread.start()
        logger.info("start short video thread successfully")

        self.is_recording = False

        def terminate_handler(sig, frame):
            logger.error("[Subprocess:Recognition]: recv SIGINT signal")
            sys.exit(0)

        signal.signal(signal.SIGINT, terminate_handler)


        # 视频处理相关
        self.cap = None
        self.out = None
        self.video_writer = None
        
    def start_recognition(self, task_data: Dict[str, Any]):
        """
        开始视频识别任务
        
        Args:
            task_data: 任务数据，包含拉流地址、推流地址等信息
        """
        if self.is_recognition_running:
            logger.warning("视频识别任务已经在运行中")
            self.send_callback_response("error", task_data, "已有任务正在运行")
            return False
        
        try:
            # 保存当前任务信息
            self.current_task = task_data
            
            # 尝试打开视频流
            pull_uri = task_data['payload']['origin_uri']
            logger.info(f"尝试打开视频流: {pull_uri}")
            
            self.cap = cv2.VideoCapture(pull_uri)
            if not self.cap.isOpened():
                error_msg = f"无法打开视频流: {pull_uri}"
                logger.error(error_msg)
                self.send_callback_response("error", task_data, error_msg)
                return False
            
            # 获取视频属性
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 初始化推流
            labeled_uri = task_data['payload']['labeled_uri']
            fourcc = cv2.VideoWriter_fourcc(*'X264')
            self.out = cv2.VideoWriter(labeled_uri, fourcc, self.fps, (width, height))
            
            if not self.out.isOpened():
                error_msg = f"无法初始化推流: {labeled_uri}"
                logger.error(error_msg)
                self.send_callback_response("error", task_data, error_msg)
                return False
            
            # 启动识别线程
            self.is_recognition_running = True
            self.should_stop = False
            
            self.recognition_thread = threading.Thread(
                target=self._recognition_worker,
                daemon=True
            )
            self.recognition_thread.start()
            logger.info("视频识别线程已启动")
            
            # 发送成功响应
            self.send_callback_response("success", task_data)
            return True
            
        except Exception as e:
            error_msg = f"启动识别任务时发生错误: {str(e)}"
            logger.error(error_msg)
            self.send_callback_response("error", task_data, error_msg)
            return False
    
    def stop_recognition(self, task_data: Dict[str, Any]):
        """
        停止视频识别任务
        
        Args:
            task_data: 任务数据
        """
        if not self.is_recognition_running:
            logger.warning("视频识别任务未在运行")
            self.send_callback_response("error", task_data, "没有正在运行的任务")
            return False
        
        try:
            self.should_stop = True
            self.is_recognition_running = False
            
            if self.recognition_thread and self.recognition_thread.is_alive():
                self.recognition_thread.join(timeout=5.0)
            
            # 释放资源
            if self.cap:
                self.cap.release()
            if self.out:
                self.out.release()
            
            logger.info("视频识别任务已停止")
            
            # 发送完成响应
            self.send_callback_response("complete", task_data)
            return True
            
        except Exception as e:
            error_msg = f"停止识别任务时发生错误: {str(e)}"
            logger.error(error_msg)
            self.send_callback_response("error", task_data, error_msg)
            return False
    
    def _recognition_worker(self):
        """
        视频识别工作线程
        """
        logger.info("视频识别工作线程开始运行")
        frame_count = 0
        detected_objects = []
        
        try:
            while not self.should_stop and self.is_recognition_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("无法读取视频帧，可能视频流已结束")
                    break
                
                # 处理视频帧（这里应该是实际的识别逻辑）
                processed_frame, detection_results = self.process_frame(frame, frame_count)
                
                # 记录检测到的目标
                if detection_results:
                    logger.info("检测到目标，开始写文件")
                    self.save_video_clip(frame_count, detection_results, processed_frame)
                else:
                    self.save_video_clip(frame_count, detection_results, processed_frame)   
                
                    # detected_objects.extend(detection_results)
                    
                    # 如果检测到特定目标，保存视频片段
                    # for result in detection_results:
                    #     if result.get('target_class') in ['car', 'human', 'landslit']:
                    #         self.save_video_clip(frame_count, result)
                
                # 推流处理后的帧
                if self.out:
                    self.out.write(processed_frame)
                
                frame_count += 1
                
                # 控制处理速度
                # time.sleep(0.03)  # 约30fps
        
        except Exception as e:
            logger.error(f"视频识别处理错误: {str(e)}")
            if self.current_task:
                self.send_callback_response("error", self.current_task, str(e))
        finally:
            logger.info("视频识别工作线程结束")
            self.is_recognition_running = False
    
    def get_position_and_attitude(self, airport_sn: str, vehicle_sn: str) -> Optional[Dict[str, Any]]:
        """
        获取无人机位置和姿态信息
        
        Args:
            airport_sn: 机场序列号
            vehicle_sn: 飞行器序列号
            
        Returns:
            Dict: 包含位置和姿态信息的字典，如果键不存在则返回 None
        """
        # 构建 Redis 键名
        key = f"cloud_uav:airport:{airport_sn}:vehicle:{vehicle_sn}:position_and_attitude"
        try:
            # 获取键的值
            value = self.redis_client.get(key)
            
            if value is None:
                print(f"键不存在: {key}")
                return None
            
            # 解析 JSON 数据
            data = json.loads(value)
            return data
            
        except redis.RedisError as e:
            print(f"Redis 错误: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {e}")
            return None

    def caculate_target_position(self, vehicle_info):
        """
        计算目标位置
        
        Args:
            vehicle_info: 无人机信息
            
        Returns:
            tuple: (处理后目标经纬)
        """
        target_latitude = vehicle_info["vehicle"].get("latitude")
        target_longitude= vehicle_info["vehicle"].get("longitude")
        target_altitude = vehicle_info["vehicle"].get("altitude")
        return target_latitude, target_longitude, target_altitude

    def process_frame(self, frame, frame_count):
        """
        处理视频帧（实际识别逻辑）
        
        Args:
            frame: 视频帧
            frame_count: 帧计数器
            
        Returns:
            tuple: (处理后的帧, 检测结果列表)
        """
        # 这里应该是实际的识别算法
        # 例如使用YOLO、SSD等目标检测模型
        # 固定经纬度（你提供的）
        # LAT = 25.070270
        # LON = 102.684488
        # ALT = 1920.12

        # 飞行器信息（从 Redis 或其他来源获取）
        vehicle_info = self.get_position_and_attitude(self.current_task["payload"].get("airport_sn"),self.current_task["payload"].get("vehicle_sn"))
        if vehicle_info:
            self.vehicle_info = vehicle_info
            lat, lon, alt = self.caculate_target_position(self.vehicle_info)
        else:
            lat, lon, alt = 41, 116, 100
        # VEHICLE_INFO = {
        #     "vehicle": {
        #         "roll": 30,
        #         "pitch": 45,
        #         "yaw": 60,
        #         "latitude": LAT,
        #         "longitude": LON,
        #         "altitude": ALT
        #     },
        #     "gimbal": {
        #         "roll": 30,
        #         "pitch": 45,
        #         "yaw": 60
        #     }
        # }

        
        # MSG_ID = "cd5925d0-8983-4445-bc9b-8700da3a5820"
        # PULL_URI = "rtsp://127.0.0.1:554/live1"
        # LABELED_URI = "rtsp://127.0.0.1:554/live1/labeled"
        # AIRPORT_SN = "xxxxxx"
        # VEHICLE_SN = "yyyyyy"
        # TENANT_ID = "aaaaaa"
        # PLAN_ID = "bbbbbb"
        # TASK_ID = "cccccc"

        # 模拟识别结果
        detection_results = []
        if frame_count % 10 == 0:  # 每10帧模拟一次检测

            detection_results = self.recon.recognize_frame(
                frame=frame,
                lat=lat,
                lon=lon,
                alt=alt,
                vehicle_info=vehicle_info,
                msg_id=self.current_task["msg_id"],
                pull_uri=self.current_task["payload"].get("origin_uri"),
                labeled_uri=self.current_task["payload"].get("labeled_uri"),
                airport_sn=self.current_task["payload"].get("airport_sn"),
                vehicle_sn=self.current_task["payload"].get("vehicle_sn"),
                tenant_id=self.current_task["payload"].get("tenant_id"),
                plan_id=self.current_task["payload"].get("plan_id"),
                task_id=self.current_task["payload"].get("task_id"),
                frame_ts_ms=int(time.time() * 1000)
            )


            # detection_results = [{
            #     "target_class": "car",
            #     "frame_ts": int(time.time() * 1000),
            #     "box": {"x_1": 60, "y_1": 60, "x_2": 120, "y_2": 120},
            #     "position": {
            #         "latitude": 25.070270,
            #         "longitude": 102.684488,
            #         "altitude": 1920.12
            #     },
            #     "conf": 0.98,
            #     "track_id": 1,
            #     "vehicle_info": {
            #         "vehicle": {
            #             "roll": 30,
            #             "pitch": 45,
            #             "yaw": 60,
            #             "latitude": 25.070270,
            #             "longitude": 102.684488,
            #             "altitude": 1920.12,
            #         },
            #         "gimbal": {
            #             "roll": 30,
            #             "pitch": 45,
            #             "yaw": 60
            #         }
            #     }
            # }]
        
        # 在帧上绘制检测框（模拟）
        processed_frame = frame.copy()
        for result in detection_results:
            logger.info(f"识别结果:{result}")
            box = result['box']
            cv2.rectangle(processed_frame, 
                         (box['x_1'], box['y_1']), 
                         (box['x_2'], box['y_2']), 
                         (0, 255, 0), 2)
            cv2.putText(processed_frame, 
                       f"{result['target_class']} {result['conf']:.2f}", 
                       (box['x_1'], box['y_1'] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return processed_frame, detection_results
    
    def write_short_video(self, rx: queue.Queue, stop_event: threading.Event):
        """写短视频的子进程

        Parameters
        ----
        rx : queue.Queue
            线程通信队列
        stop_event : threading.Event
            线程停止事件
        """
        video_writer = None
        cover_data = None
        cover_path = None
        while not stop_event.is_set():
            frame_msg = rx.get()

            operate = frame_msg["operate"]
            if operate == "terminate":
                return
            elif operate == "open":
                video_path: Path = frame_msg["video_path"]
                fourcc = cv2.VideoWriter.fourcc(*"MP4V")  ## need to modify by X264
                fps = frame_msg["fps"]
                frame_size = frame_msg["frame_size"]
                video_path_str = str(video_path)
                cover_path = video_path_str.replace(".mp4", ".jpg")
                video_writer = cv2.VideoWriter(video_path_str, fourcc, fps, frame_size)
            elif operate == "close":
                video_writer.release()

                video_writer = None
                cover_data = None
            else:
                if cover_data is None:
                    cover_data = frame_msg["data"]
                    # 写入第一帧作为封面
                    logger.info("prepare to write cover, {}", cover_path)
                    cv2.imwrite(cover_path, cover_data)
                video_writer.write(frame_msg["data"])
        logger.warning("Thread:ShortVideoWriter stopped, clearing")
        if video_writer is not None:
            video_writer.release()
        logger.warning("Thread:ShortVideoWriter stopped, cleared")


    def save_video_clip(self, current_frme_index, detection_results, frame ):
        """
        保存检测到目标时的视频片段（前7秒+后7秒）
        
        Args:
            current_frame_index: 当前帧索引
            detection_result: 检测结果
        """
        # 这里应该是实际的视频片段保存逻辑
        # 需要实现保存前7秒和后7秒的视频
        
        
        
        # 创建保存目录
        os.makedirs("video_clips", exist_ok=True)

        if not self.is_recording:
            logger.info(f"检测到目标消息 {self.current_task.get('msg_id')}，保存视频片段")
            self.save_res = detection_results
            self.recorded_count = 0
            self.recorded_max = self.fps *15
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_filename = f"video_clips/{timestamp}_{self.current_task.get('msg_id')}.mp4"
            
            # 这里应该是实际的视频保存逻辑
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30  # 获取帧率，如果获取失败则默认30
            frame_size = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            
            self.short_video_queue.put_nowait(
                        {
                            "operate": "open",
                            "video_path": self.save_filename,
                            "fps": fps,
                            "frame_size": frame_size,
                        }
                    )
            self.is_recording = True
        else:
            self.short_video_queue.put_nowait({"operate": "write", "data": frame})
            self.recorded_count += 1
            # print("recorded_count:",self.recorded_count)
            if self.recorded_count > self.recorded_max:
                self.short_video_queue.put_nowait({"operate": "close"})
                self.is_recording = False
                self.send_recognition_result(self.save_res, self.save_filename)
                self.recorded_count = 0
                logger.info(f"视频片段已保存: {self.save_filename}")

        # 由于实现完整的前后7秒视频保存较复杂，这里只做演示
        
        # return filename
        # detection_result["video_path"] = filename
        # 发送识别结果
        
    
    def send_callback_response(self, callback_type: str, task_data: Dict[str, Any], error_message: str = None):
        """
        发送回调响应到Redis
        
        Args:
            callback_type: 回调类型 (success, error, complete)
            task_data: 任务数据
            error_message: 错误消息（可选）
        """
        response = {
            "callback_type": callback_type,
            "msg_id": task_data.get("msg_id", str(uuid.uuid4())),
            "ts": int(time.time()),
            "payload": task_data.get("payload", {}),
            "error_message": error_message
        }
        
        try:
            self.redis_client.publish(self.callback_channel, json.dumps(response))
            logger.info(f"已发送 {callback_type} 响应到 {self.callback_channel}")
        except Exception as e:
            logger.error(f"发送回调响应时发生错误: {str(e)}")
    
    def send_recognition_result(self, results: List[Dict[str, Any]], filename):
        """
        发送识别结果到Redis
        
        Args:
            results: 识别结果列表
        """
        if not self.current_task:
            logger.warning("没有当前任务信息，无法发送识别结果")
            return
        
        # 构建结果消息
        result_message = {
            "video_path": filename,  # 实际应该替换为保存的视频路径
            "ts": int(time.time()),
            "meta_info": {
                "msg_id": self.current_task.get("msg_id"),
                "pull_uri": self.current_task["payload"].get("origin_uri"),
                "labeled_uri": self.current_task["payload"].get("labeled_uri"),
                "airport_sn": self.current_task["payload"].get("airport_sn"),
                "vehicle_sn": self.current_task["payload"].get("vehicle_sn"),
                "tenant_id": self.current_task["payload"].get("tenant_id"),
                "plan_id": self.current_task["payload"].get("plan_id"),
                "task_id": self.current_task["payload"].get("task_id")
            },
            "results": results
        }
        
        try:
            self.redis_client.publish(self.result_channel, json.dumps(result_message))
            logger.info(f"已发送识别结果到 {self.result_channel}")
        except Exception as e:
            logger.error(f"发送识别结果时发生错误: {str(e)}")
    
    def process_message(self, message: Dict[str, Any]):
        """
        处理接收到的Redis消息
        
        Args:
            message: Redis消息
        """
        try:
            if message['type'] != 'message':
                return
            
            data_str = message['data']
            logger.info(f"接收到消息: {data_str}")
            
            # 解析JSON数据
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError as e:
                logger.error(f"消息JSON解析错误: {str(e)}")
                return
            
            # 检查消息类型
            msg_type = data.get("msg_type")
            if not msg_type:
                logger.warning("消息缺少 msg_type 字段")
                return
            
            # 处理开始识别消息
            if msg_type == "start":
                logger.info("收到开始识别消息")
                self.start_recognition(data)
            
            # 处理停止识别消息
            elif msg_type == "stop":
                logger.info("收到停止识别消息")
                self.stop_recognition(data)
            
            else:
                logger.warning(f"未知的消息类型: {msg_type}")
                
        except Exception as e:
            logger.error(f"处理消息时发生错误: {str(e)}")
    
    def start_listening(self):
        """
        开始监听Redis频道
        """
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(self.signal_channel)
            
            logger.info(f"开始监听频道: {self.signal_channel}")
            logger.info("等待开始/停止识别消息...")
            
            # 监听消息
            for message in pubsub.listen():
                if message['type'] == 'subscribe':
                    continue
                
                self.process_message(message)
                
        except Exception as e:
            logger.error(f"监听过程中发生错误: {str(e)}")
        finally:
            # 确保停止识别任务
            if self.is_recognition_running and self.current_task:
                self.stop_recognition(self.current_task)
    
    def cleanup(self):
        """
        清理资源
        """
        if self.is_recognition_running and self.current_task:
            self.stop_recognition(self.current_task)
        
        if self.redis_client:
            self.redis_client.close()

def main():
    """主函数"""
    # 创建视频识别服务实例
    service = VideoRecognitionService(
        host='thzy-001.redis.rds.aliyuncs.com',      # Redis服务器地址
        port=6379,             # Redis端口
        password='Sonic513',         # Redis密码（如果有）
        db=0                   # Redis数据库编号
    )
    
    try:
        logger.info("启动视频识别服务...")
        service.start_listening()
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
    finally:
        service.cleanup()

if __name__ == "__main__":
    main()