import signal
import threading
import time
from typing import Dict, Tuple, Mapping
import cv2
import asyncio
from ultralytics import YOLO
import subprocess
import uuid
import os
import traceback
from loguru import logger
from multiprocessing import Queue
import json
import redis
from threading import Thread
import queue
import global_vars
import sys
from pathlib import Path

sub_procs = []

# 是否启用调试模式，启用调试模式后：1. 不与 Redis 进行任何操作；2. 不与消息队列中间件进行任何操作; 3. 不构造任何ResultMessage
debug_mode = False
if not debug_mode:
    from result_message import (
        ResultMessageMedia,
        ResultMessage,
        ResultMessageMedia,
        ResultMessageFrom,
    )


def write_short_video(rx: queue.Queue, stop_event: threading.Event):
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
            fourcc = cv2.VideoWriter.fourcc(*"X264")
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


def load_models(pts: Mapping[str, str]) -> Mapping[str, YOLO]:
    """加载 YOLO 权重文件

    Parameters
    ----
    pts : Dict[str, str]
        权重文件

        该参数的类型为字典，其键应当为权重名称，其值应当为模型文件路径。如：{"animal": "./weights/animal_det.pt", "fire": "./weights/fire_det.pt"}

    Returns
    -----
    Mapping[str, YOLO]
        模型权重文件

        其键应当为权重名称，其值应当为 YOLO 模型对象。
    """
    models = {
        "animal": YOLO(pts["animal"]),
        # "building": YOLO(pts["building"]),
        # "fire": YOLO(pts["fire"])
    }
    return models


async def read_stream(pipe, prefix):
    while True:
        logger.info("read stream")
        line = await pipe.readline()
        if not line:
            break
        logger.info(f"{prefix}: {line.decode().strip()}")


def get_vehicle_position(redis_ctx: redis.Redis) -> tuple[float, float]:
    """从 Redis 中获取无人机的位置

    从 Redis 中获取无人机的当前位置，Redis 中保存的键为 `vehicle.position`，其值示例如下：
    ```json
    {
        "latitude_deg": 1.23,
        "longitude_deg": 1.23,
        "relative_altitude_m": 0.1,
        "absolute_altitude_m": 100.1,
    }
    ```
    如果位置信息反序列化失败，则返回 (0.0, 0.0)

    Parameters
    ----
    redis_ctx : redis.Redis
        Redis 客户端

    Returns
    ----
    Tuple[float, float]
        无人机当前所处的纬度和经度（如发生错误，则范围(0.0, 0.0)）
    """
    position = redis_ctx.get("vehicle:position")
    if position is None:
        return 0.0, 0.0
    try:
        position = json.loads(position)
    except json.decoder.JSONDecodeError:
        logger.error("failed to decode vehicle.position")
        return 0.0, 0.0
    return position["latitude_deg"], position["longitude_deg"]


def recognition(
    task_id: str,
    in_uri: str,
    out_uri: str,
    pts: Mapping[str, str],
    mq_queue: Queue,
    ffmpeg_path: str,
    data_path: Path,
    redis_config: Mapping[str, str | int],
    device_id: str,
):
    """识别视频流

    该函数从 RTSP 服务器读取 `in_uri` 指定的视频流，加载 `pts` 指定的模型，对视频流进行逐帧识别后，对视频流进行标记后输出到 `out_uri` 指定
    的视频流。

    Parameters
    ----
    task_id : str
        任务编号
    in_uri : str
        输入视频流

        RTSP 视频流。该函数将使用 opencv 库的 VideoCapture 从该视频流读取视频数据，并进行后续的操作。视频流地址示例：rtsp://127.0.0.1:554/vide1
    out_uri : str
        输出的视频流

        在对视频进行识别等处理后，将在视频帧上对结果进行标记，随后将视频帧输出到该参数指定的视频流，例如：rtsp://127.0.0.1:554/vide1/labeled
    pts : Mapping[str, str]
        模型权重文件
    mq_queue : Queue
        一个多进程队列

        在该函数中，将写入识别或处理结果到该队列中，在另一进程中将从该队列中读取结果数据，并将其发送到消息队列中间件中
    ffmpeg_path : str
        FFMPEG 可执行文件路径
    data_path : str
        短视频保存路径（不包含任务编号）

    redis_config : Mapping[str, str]
        Redis 配置

        至少应当包含Redis 主机和 Redis 端口。示例：{"host": "127.0.0.1", "port": 6379}
    """
    logger.info(f"start recognition video stream, input: {in_uri}, output: {out_uri}")
    models = load_models(pts)

    logger.info("prepare to initialize the redis client...")
    # 初始化 Redis 客户端
    redis_ctx: redis.Redis = redis.Redis(
        host=redis_config["host"], port=redis_config["port"]
    )
    logger.info("initialize the redis client successfully")

    result_from = ResultMessageFrom("uav", device_id)

    logger.info(f"start to read the video stream, input: {in_uri}")
    cap = cv2.VideoCapture(in_uri)
    if not cap.isOpened():
        logger.error(f"failed to open video stream: {in_uri}")
        return

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # 获取帧率，如果获取失败则默认30
        frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        # 确保帧尺寸有效，如果获取失败则设置默认值
        if frame_size == (0, 0):
            frame_size = (640, 480)
            logger.warning(
                "Failed to get frame size from video stream, using default size 640x480."
            )
        logger.info(f"video stream information: fps={fps}, frame_size={frame_size}")
        logger.info("prepare to start start ffmpeg stream push process...")
        ffmpeg_cmd = [
            ffmpeg_path,  # 你的 ffmpeg 路径
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{frame_size[0]}x{frame_size[1]}",
            "-r",
            f"{fps}",
            "-i",
            "-",  # 输入来自管道
            "-c:v",
            "libx264",
            "-tune",
            "zerolatency",
            # '-flush_packets', '1',  # 添加这一行
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-f",
            "rtsp",
            out_uri,
        ]

        ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        ffmpeg_process.name = "ffmpeg_process"
        global_vars.procs.append(ffmpeg_process)
        event_loop = asyncio.get_event_loop()
        logger.info("ffmpeg stream push process started")
        # event_loop.create_task(read_stream(
        #     ffmpeg_process.stdout, "FFMPEG STDOUT"))
        # event_loop.create_task(read_stream(
        #     ffmpeg_process.stderr, "FFMPEG STDERR"))

        thread_stop_event = threading.Event()
        task_dir = data_path.joinpath(task_id)
        logger.info(
            f"prepare to start a subthread to write short video, task_dir: {task_dir}"
        )
        short_video_queue = queue.Queue()
        short_video_thread = Thread(
            target=write_short_video, args=(short_video_queue, thread_stop_event)
        )
        short_video_thread.start()
        logger.info("start short video thread successfully")

        def terminate_handler(sig, frame):
            logger.error("[Subprocess:Recognition]: recv SIGINT signal")
            # 结束线程
            thread_stop_event.set()
            for proc in sub_procs:
                proc.stdin.write(b"q\n")
                proc.terminate()
                proc.wait()
            cap.release()
            sys.exit(0)

        signal.signal(signal.SIGINT, terminate_handler)

        cnt = 0

        os.makedirs(task_dir, exist_ok=True)
        current_hit: bool = False  # 当前帧是否命中
        recording: bool = False  # 是否正在录制
        max_record_count: int = fps * 15
        recorded_count: int = 0  # 录制的帧数
        media_id: str | None = None  # 结果编号
        media_file_path: str | None = None  # 结果目录
        media_ts: int | None = None
        result_message_media: ResultMessageMedia | None = None

        logger.info("start handle video stream...")
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error(f"Failed to read frame from video stream.")
                break

            results = {}

            # logger.info("prepare to get vehicle position...")
            vehicle_position: Tuple[float, float] = (0.0, 0.0)
            if not debug_mode:
                # 获取无人机的位置
                vehicle_position = get_vehicle_position(redis_ctx)
                # logger.info(f"current vehicle_position: {vehicle_position}")
            # logger.info("get vehicle position successfully")
            # logger.info("prepare to detect target use yolo...")
            for model_name, model in models.items():
                result_list = model(frame, verbose=False)  # 使用 verbose 来抑制输出
                results[model_name] = result_list
            # logger.info("detect target use yolo successfully")
            sheep_count = 0
            cattle_count = 0
            frame_ts = int(time.time_ns() / 1e6)
            for model_name, result_list in results.items():
                for r in result_list:
                    boxes = r.boxes.cpu().numpy()
                    # 判断是否检测到
                    if len(boxes.xyxy) != 0:
                        # logger.info(f"detected target, count: {len(boxes.xyxy)}")
                        current_hit = True
                    for box in boxes:
                        result_id = str(uuid.uuid4())
                        if len(box.xyxy) == 0:
                            continue
                        r_cls = int(box.cls[0])
                        r_conf = float(box.conf[0])

                        # 可能的值: flame sheep cattle Building Road Water Tree Grass
                        class_name = (
                            models[model_name].names.get(r_cls, "Unknown")
                            if hasattr(models[model_name], "names")
                            else "Unknown"
                        )
                        r_box = box.xyxy[0].astype(int)
                        cv2.rectangle(
                            frame,
                            (r_box[0], r_box[1]),
                            (r_box[2], r_box[3]),
                            (0, 255, 0),
                            2,
                        )
                        label = f"{class_name} {r_conf:.2f}"
                        cv2.putText(
                            frame,
                            label,
                            (r_box[0], r_box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                        if not debug_mode:
                            if class_name == "Sheep":
                                sheep_count += 1
                            elif class_name == "Cattle":
                                cattle_count += 1
                            else:
                                # 封装结果
                                result_message = ResultMessage(
                                    "result",
                                    task_id,
                                    result_id,
                                    class_name,
                                    frame_ts,
                                    vehicle_position[0],
                                    vehicle_position[1],
                                    result_message_media,
                                    result_from,
                                )
                                if result_message.media is not None:
                                    _media_path = Path(result_message.media.path)
                                    if _media_path.is_absolute():
                                        _media_path = _media_path.relative_to(task_dir)
                                    result_message.media.path = str(_media_path)
                                mq_queue.put_nowait(
                                    {"type": "result", "data": result_message}
                                )

            # 封装动物数量消息
            if sheep_count != 0 or cattle_count != 0:
                animal_count_message = ResultMessage(
                    "animal_count",
                    task_id,
                    str(uuid.uuid4()),
                    "animal_count",
                    frame_ts,
                    vehicle_position[0],
                    vehicle_position[1],
                    result_message_media,
                    result_from,
                    [
                        {"label": "sheep", "count": sheep_count},
                        {"label": "cattle", "count": cattle_count},
                    ],
                )
                mq_queue.put_nowait({"type": "result", "data": animal_count_message})
            # 开始录制
            if recording == False and current_hit == True:
                recording = True
                # 创建短视频保存路径
                media_id = str(uuid.uuid4())
                media_file_path = task_dir.joinpath(f"{media_id}.mp4")
                media_ts = int(time.time_ns() / 1e6)
                short_video_queue.put_nowait(
                    {
                        "operate": "open",
                        "video_path": media_file_path,
                        "fps": fps,
                        "frame_size": frame_size,
                    }
                )
                recorded_count += 1
                if not debug_mode:
                    result_message_media = ResultMessageMedia(
                        media_id, media_ts, str(media_file_path), task_id
                    )
            # 正在录制
            if recording:
                short_video_queue.put_nowait({"operate": "write", "data": frame})
                recorded_count += 1
            # 结束录制
            if recorded_count >= max_record_count:
                short_video_queue.put_nowait({"operate": "close"})
                recording = False
                recorded_count = 0
                # 修改媒体路径为相对路径
                _media_path = Path(result_message_media.path)
                if _media_path.is_absolute():
                    _media_path = _media_path.relative_to(task_dir)
                result_message_media.path = str(_media_path)
                mq_queue.put_nowait({"type": "media", "data": result_message_media})
            # logger.info("prepare to write frame to ffmpeg process...")
            ffmpeg_process.stdin.write(frame.tobytes())
            # logger.info("write frame to ffmpeg process successfully.")
            cnt += 1
            prev_hit = current_hit

    except Exception as e:
        logger.error(f"Some errors occurred in recognition, error={e}")
        traceback.print_exc()
    finally:
        cap.release()
        if (
            "ffmpeg_process" in locals() and ffmpeg_process
        ):  # 确保 ffmpeg_process 被定义且不为空
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()


if __name__ == "__main__":
    in_uri = "rtsp://127.0.0.1:8554/video1"  # RTSP 拉流地址6
    out_uri = "rtsp://127.0.0.1:8554/output_stream"  # RTSP 推流地址
    pts = {
        "animal": "./weights/cattle_sheep_det.pt",
    }
    task_id = str(uuid.uuid4())
    requests_queue = None
    data_path = Path("./results/videos/")
    ffmpeg_path = "D:/opt/ffmpeg/bin/ffmpeg.exe"
    redis_config = {
        "host": "127.0.0.1",  # Redis 主机地址
        "port": 6379,  # Redis 端口号
    }
    device_id = "uav_device_001"  # 设备ID
    asyncio.run(
        recognition(
            task_id=task_id,
            in_uri=in_uri,
            out_uri=out_uri,
            pts=pts,
            mq_queue=requests_queue,
            ffmpeg_path=ffmpeg_path,
            data_path=data_path,
            redis_config=redis_config,  # 添加 redis_config
            device_id=device_id,  # 添加 device_id
        )
    )
