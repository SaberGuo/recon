import redis
import signal
import json
import threading
from threading import Thread
import logging
import time
import cv2
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from recognizer import UAVRecognizer
import os
from datetime import datetime
import queue
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('VideoRecognitionManager')

# 枚举定义
class SignalMsgType(Enum):
    START = "start"
    STOP = "stop"


CallbackType_ERROR = "error"
CallbackType_SUCCESS = "success"
CallbackType_COMPLETE = "complete"

class CallbackType(Enum):
    SUCCESS = "success"
    ERROR = "error"
    COMPLETE = "complete"

# 数据结构定义
@dataclass
class Payload:
    origin_uri: str
    labeled_uri: str
    airport_sn: str
    vehicle_sn: str
    tenant_id: str
    plan_id: str
    task_id: str

@dataclass
class SignalMessage:
    msg_type: SignalMsgType
    msg_id: str
    ts: int
    payload: Payload

@dataclass
class CallbackMessage:
    callback_type: str
    msg_id: str
    ts: int
    payload: Payload
    error_message: Optional[str] = None

@dataclass
class DetectionResult:
    target_class: str
    frame_ts: int
    box: Dict[str, int]
    position: Dict[str, float]
    conf: float
    track_id: Optional[int] = None
    vehicle_info: Optional[Dict[str, Any]] = None

@dataclass
class ResultMessage:
    video_path: str
    ts: int
    meta_info: Dict[str, Any]
    results: List[DetectionResult]

class VideoProcessingTask:
    """单个视频处理任务"""
    
    def __init__(self, signal_msg: SignalMessage, redis_client):
        self.redis_client = redis_client
        self.signal_msg = signal_msg
        self.task_id = signal_msg.payload.task_id
        self.is_running = False
        self.should_stop = False
        self.thread = None
        # self.current_task: Optional[Dict[str, Any]] = None

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
        
        # 识别后 video 记录
        self.cap = None
        self.out = None
        self.video_writer = None
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

        
    def start(self):
        """启动任务处理线程"""
        if self.is_running:
            logger.warning(f"任务 {self.task_id} 已经在运行中")
            return False
            
        self.is_running = True
        self.should_stop = False
        
        self.thread = threading.Thread(target=self._process_video, daemon=True)
        self.thread.start()
        logger.info(f"视频处理任务 {self.task_id} 已启动")
        return True
        
    def stop(self):
        """停止任务处理线程"""
        if not self.is_running:
            logger.warning(f"任务 {self.task_id} 未在运行")
            return False
            
        self.should_stop = True
        self.is_running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10.0)
            logger.info(f"视频处理任务 {self.task_id} 已停止")
        
        # 释放资源
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        
        if self.redis_client:
            self.redis_client.close()
        return True
        
    def _process_video(self):
        """视频处理主逻辑"""
        logger.info(f"开始处理视频任务 {self.task_id}")
        
        try:
            # 1. 从媒体服务器拉取视频流
            if not self._open_video_stream():
                error_msg = f"无法打开视频流: {self.signal_msg.payload.origin_uri}"
                self._send_callback(CallbackType_ERROR, error_msg)
                return
                
            # 2. 发送成功开始回调
            self._send_callback(CallbackType_SUCCESS)
            
            # 3. 处理视频帧
            frame_count = 0
            while not self.should_stop and self.is_running:
                try:
                    # 获取视频帧
                    frame = self._get_next_frame()
                    if frame is None:
                        break
                        
                    # 处理帧（识别、标记等）
                    processed_frame, detection_results = self._process_frame(frame, frame_count)
                    logger.info(f"处理完成，获得检测结果：{detection_results}")
                    # 推送到媒体服务器
                    self._push_processed_frame(processed_frame)
                    
                    # 如果有检测结果，保存短视频片段并发送结果
                    if detection_results:
                        self._handle_detection_results(detection_results, frame_count)
                    
                    frame_count += 1
                    
                except Exception as e:
                    logger.error(f"处理视频帧时发生错误: {e}")
                    time.sleep(0.1)  # 短暂等待后继续
                    
        except Exception as e:
            logger.error(f"视频处理任务异常: {e}")
            self._send_callback(CallbackType_ERROR, str(e))
        finally:
            # 清理资源
            self._close_video_stream()
            logger.info(f"视频处理任务 {self.task_id} 结束")
            
    def _open_video_stream(self) -> bool:
        """打开视频流"""
        # 这里实现打开视频流的逻辑
        # 使用OpenCV、FFmpeg或其他库
        pull_uri = self.signal_msg.payload.origin_uri
        logger.info(f"尝试打开视频流: {pull_uri}")
        self.cap = cv2.VideoCapture(pull_uri)
        if not self.cap.isOpened():
            error_msg = f"无法打开视频流: {pull_uri}"
            logger.error(error_msg)
            # self._send_callback()
            self._send_callback(CallbackType_ERROR, error_msg)
            #self.send_callback_response("error", self.signal_msg, error_msg)
            return False

        # 获取视频属性
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

         # 初始化推流
        labeled_uri = self.signal_msg.payload.labeled_uri
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        self.out = cv2.VideoWriter(labeled_uri, fourcc, self.fps, (width, height))
        
        logger.info(f"尝试打开视频流: {self.signal_msg.payload.origin_uri}")
        if not self.out.isOpened():
            error_msg = f"无法初始化推流: {labeled_uri}"
            logger.error(error_msg)
            self._send_callback(CallbackType_ERROR,  error_msg)
            return False
        # 模拟打开视频流
        # time.sleep(1)  # 模拟连接时间
        
        # 模拟成功或失败
        return True  # 或 False 如果连接失败
        
    def _close_video_stream(self):
        """关闭视频流"""
        # 这里实现关闭视频流的逻辑
        logger.info(f"关闭视频流: {self.signal_msg.payload.origin_uri}")
        
    def _get_next_frame(self):
        """获取下一帧"""
        # 这里实现获取视频帧的逻辑
        # 模拟获取帧
        if self.should_stop:
            return None
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("无法读取视频帧，可能视频流已结束")
            return None
        return frame
        # time.sleep(0.033)  # 模拟30fps
        # return f"frame_{int(time.time() * 1000)}"
    
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

    def _save_video_clip(self, current_frme_index, detection_results, frame ):
        """
        保存检测到目标时的视频片段（前7秒+后7秒）
        
        Args:
            current_frame_index: 当前帧索引
            detection_result: 检测结果
        """
        # 这里应该是实际的视频片段保存逻辑
        # 需要实现保存前7秒和后7秒的视频

        if not self.is_recording and detection_results:
            # 创建保存目录
            os.makedirs("video_clips", exist_ok=True)
            logger.info(f"检测到目标消息 {self.signal_msg.msg_id}，保存视频片段")
            self.save_res = detection_results
            self.recorded_count = 0
            self.recorded_max = self.fps *15
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_filename = f"video_clips/{timestamp}_{self.signal_msg.payload.task_id}.mp4"
            
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
        elif self.is_recording:
            self.short_video_queue.put_nowait({"operate": "write", "data": frame})
            self.recorded_count += 1
            # print("recorded_count:",self.recorded_count)
            if self.recorded_count > self.recorded_max:
                self.short_video_queue.put_nowait({"operate": "close"})
                self.is_recording = False
                self._handle_detection_results(self.save_res, self.save_filename)
                self.recorded_count = 0
                logger.info(f"视频片段已保存: {self.save_filename}")
        
        # return filename
        # detection_result["video_path"] = filename
        # 发送识别结果

    def _process_frame(self, frame, frame_count):
        """处理帧（识别、标记等）"""
        # 这里实现帧处理和识别逻辑
        # 模拟处理
        # processed_frame = f"processed_{frame}"
        
        # 模拟检测结果（每30帧检测到一次目标）
        detection_results = []
        if frame_count % 10 == 0:
            # 飞行器信息（从 Redis 或其他来源获取）
            vehicle_info = self._get_position_and_attitude(self.signal_msg.payload.airport_sn,self.signal_msg.payload.vehicle_sn)
            if vehicle_info:
                self.vehicle_info = vehicle_info
                lat, lon, alt = self._caculate_target_position(self.vehicle_info)
            else:
                lat, lon, alt = 41, 116, 100

            detection_results = self.recon.recognize_frame(
                frame=frame,
                lat=lat,
                lon=lon,
                alt=alt,
                vehicle_info=vehicle_info,
                msg_id=self.signal_msg.msg_id,
                pull_uri=self.signal_msg.payload.origin_uri,
                labeled_uri=self.signal_msg.payload.labeled_uri,
                airport_sn=self.signal_msg.payload.airport_sn,
                vehicle_sn=self.signal_msg.payload.vehicle_sn,
                tenant_id=self.signal_msg.payload.tenant_id,
                plan_id=self.signal_msg.payload.plan_id,
                task_id=self.signal_msg.payload.task_id,
                frame_ts_ms=int(time.time() * 1000)
            )

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
            
            self._save_video_clip(frame_count, detection_results, processed_frame)
            # detection_results = [
            #     DetectionResult(
            #         target_class="car",
            #         frame_ts=int(time.time() * 1000),
            #         box={"x_1": 60, "y_1": 60, "x_2": 120, "y_2": 120},
            #         position={"latitude": 25.070270, "longitude": 102.684488, "altitude": 1920.12},
            #         conf=0.98,
            #         track_id=1,
            #         vehicle_info={
            #             "vehicle": {"roll": 30, "pitch": 45, "yaw": 60, 
            #                        "latitude": 25.070270, "longitude": 102.684488, "altitude": 1920.12},
            #             "gimbal": {"roll": 30, "pitch": 45, "yaw": 60}
            #         }
            #     )
            # ]
            
        return processed_frame, detection_results
        
    def _push_processed_frame(self, frame):
        """推送处理后的帧到媒体服务器"""
        # 这里实现推送帧到媒体服务器的逻辑
        # 模拟推送

        if self.out:
            self.out.write(frame)
        pass
        
    def _handle_detection_results(self, detection_results, video_path):
        """处理检测结果（保存短视频、发送结果等）"""
        # 1. 保存前7秒和后7秒的视频片段
        # video_path = self._save_video_segment(frame_count)
        
        # 2. 准备结果消息
        result_message = ResultMessage(
            video_path=video_path,
            ts=int(time.time()),
            meta_info={
                "msg_id": self.signal_msg.msg_id,
                "pull_uri": self.signal_msg.payload.origin_uri,
                "labeled_uri": self.signal_msg.payload.labeled_uri,
                "airport_sn": self.signal_msg.payload.airport_sn,
                "vehicle_sn": self.signal_msg.payload.vehicle_sn,
                "tenant_id": self.signal_msg.payload.tenant_id,
                "plan_id": self.signal_msg.payload.plan_id,
                "task_id": self.signal_msg.payload.task_id
            },
            results=detection_results
        )
        
        # 3. 发送结果到Redis
        self._send_result(result_message)
    
    def _get_position_and_attitude(self, airport_sn: str, vehicle_sn: str) -> Optional[Dict[str, Any]]:
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

    def _caculate_target_position(self, vehicle_info):
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
    
    def _save_video_segment(self, frame_count) -> str:
        """保存视频片段"""
        # 这里实现保存视频片段的逻辑
        # 模拟保存
        video_path = f"/data/videos/{self.task_id}_{int(time.time())}.mp4"
        logger.info(f"保存视频片段: {video_path}")
        return video_path
        
    def _send_callback(self, callback_type: str, error_message: Optional[str] = None):
        """发送回调消息到Redis"""
        # callback_msg = CallbackMessage(
        #     callback_type=callback_type,
        #     msg_id=self.signal_msg.msg_id,
        #     ts=int(time.time()),
        #     payload=self.signal_msg.payload,
        #     error_message=error_message
        # )
        response = {
            "callback_type": callback_type,
            "msg_id": self.signal_msg.msg_id,
            "ts": int(time.time()),
            "payload": self.signal_msg.payload.__dict__,
            "error_message": error_message
        }
        
        # 发送到Redis
        # redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        try:
            self.redis_client.publish(
                'cloud_uav:channel:recognize:callback',
                json.dumps(response)
            )
            logger.info(f"已发送回调消息: {callback_type}")
        except Exception as e:
            logger.error(f"发送回调消息时发生错误: {e}")
        finally:
            self.redis_client.close()
            
    def _send_result(self, result_message: ResultMessage):
        """发送识别结果到Redis"""
        # 发送到Redis
        
        try:
            self.redis_client.publish(
                'cloud_uav:channel:recognize:result',
                json.dumps(result_message.__dict__)
            )
            logger.info(f"已发送识别结果: {result_message.video_path}")
        except Exception as e:
            logger.error(f"发送识别结果时发生错误: {e}")

class VideoRecognitionManager:
    """视频识别管理器，处理多个识别任务"""
    
    def __init__(self, host='localhost', port=6379, password=None, db=0):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True
        )
        
        # 频道名称
        self.signal_channel = 'cloud_uav:channel:recognize:signal'
        
        # 任务管理
        self.tasks: Dict[str, VideoProcessingTask] = {}
        self.is_listening = False
        
    def start_listening(self):
        """开始监听Redis信号频道"""
        try:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(self.signal_channel)
            
            logger.info(f"开始监听信号频道: {self.signal_channel}")
            self.is_listening = True
            
            # 监听消息
            for message in pubsub.listen():
                if not self.is_listening:
                    break
                    
                if message['type'] != 'message':
                    continue
                    
                self._handle_signal_message(message)
                
        except Exception as e:
            logger.error(f"监听过程中发生错误: {e}")
        finally:
            self.is_listening = False
            
    def stop_listening(self):
        """停止监听"""
        self.is_listening = False
        logger.info("已停止监听信号频道")
        
    def stop_all_tasks(self):
        """停止所有任务"""
        for task_id in list(self.tasks.keys()):
            self._stop_task(task_id)
            
    def _handle_signal_message(self, message: Dict[str, Any]):
        """处理接收到的信号消息"""
        try:
            data = message['data']
            logger.info(f"接收到信号消息: {data}")
            
            # 解析JSON数据
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                logger.error(f"无法解析JSON消息: {data}")
                return
                
            # 转换为SignalMessage对象
            signal_msg = self._parse_signal_message(parsed_data)
            if not signal_msg:
                return
                
            # 根据消息类型处理
            if signal_msg.msg_type == SignalMsgType.START:
                self._start_task(signal_msg)
            elif signal_msg.msg_type == SignalMsgType.STOP:
                self._stop_task(signal_msg.payload.task_id)
                
        except Exception as e:
            logger.error(f"处理信号消息时发生错误: {e}")
            
    def _parse_signal_message(self, data: Dict[str, Any]) -> Optional[SignalMessage]:
        """解析信号消息"""
        try:
            # 解析消息类型
            msg_type = SignalMsgType(data.get('msg_type', ''))
            
            # 解析payload
            payload_data = data.get('payload', {})
            payload = Payload(
                origin_uri=payload_data.get('origin_uri'),
                labeled_uri=payload_data.get('labeled_uri'),
                airport_sn=payload_data.get('airport_sn'),
                vehicle_sn=payload_data.get('vehicle_sn'),
                tenant_id=payload_data.get('tenant_id'),
                plan_id=payload_data.get('plan_id'),
                task_id=payload_data.get('task_id')
            )
            
            # 创建SignalMessage对象
            signal_msg = SignalMessage(
                msg_type=msg_type,
                msg_id=data.get('msg_id', str(uuid.uuid4())),
                ts=data.get('ts', int(time.time())),
                payload=payload
            )
            
            return signal_msg
            
        except (KeyError, ValueError) as e:
            logger.error(f"解析信号消息时发生错误: {e}")
            return None
            
    def _start_task(self, signal_msg: SignalMessage):
        """启动新任务"""
        task_id = signal_msg.payload.task_id
        
        # 检查任务是否已存在
        if task_id in self.tasks:
            logger.warning(f"任务 {task_id} 已存在")
            return
            
        # 创建并启动任务
        task = VideoProcessingTask(signal_msg, self.redis_client)
        if task.start():
            self.tasks[task_id] = task
            logger.info(f"已启动任务 {task_id}")
            
    def _stop_task(self, task_id: str):
        """停止指定任务"""
        if task_id not in self.tasks:
            logger.warning(f"任务 {task_id} 不存在")
            return
            
        task = self.tasks[task_id]
        if task.stop():
            del self.tasks[task_id]
            logger.info(f"已停止任务 {task_id}")

# 主程序
if __name__ == "__main__":
    # 创建视频识别管理器
    manager = VideoRecognitionManager(
         host='thzy-001.redis.rds.aliyuncs.com',      # Redis服务器地址
        port=6379,             # Redis端口
        password='Sonic513',         # Redis密码（如果有）
        db=0                   # Redis数据库编号
    )
    
    try:
        logger.info("启动视频识别管理器...")
        manager.start_listening()
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行错误: {e}")
    finally:
        manager.stop_all_tasks()
        manager.stop_listening()