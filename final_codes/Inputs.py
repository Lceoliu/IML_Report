from typing import Generator, Optional, Union, Iterator
import cv2
import numpy as np
from abc import ABC, abstractmethod
import os
import time
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO | logging.DEBUG | logging.WARNING,
    filename='inputs.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class Inputs(ABC):
    @abstractmethod
    def get_input_stream(self) -> Union[np.ndarray, Iterator[np.ndarray], str]:
        """
        抽象方法获取输入流。
        返回类型可以是：
        - np.ndarray: 单张图片
        - Iterator[np.ndarray]: 图片流生成器
        - str: 文件路径（兼容ultralytics）
        """
        pass

    @abstractmethod
    def get_source_info(self) -> dict:
        """
        获取输入源信息
        """
        pass

    def is_stream(self) -> bool:
        """
        判断是否为流输入
        """
        return False


class WindowCaptureStreamInputs(Inputs):
    def __init__(self, window_name: str = None, region: tuple = None, fps: int = 30):
        """
        初始化窗口截图输入流

        :param window_name: 窗口名称，None表示截取整个屏幕
        :param region: 截取区域 (x, y, width, height)
        :param fps: 截图帧率
        """
        self.window_name = window_name
        self.region = region
        self.fps = fps
        self.frame_interval = 1.0 / fps
        self._running = False

    def get_input_stream(self) -> Iterator[np.ndarray]:
        """
        获取窗口截图流
        """
        self._running = True
        last_time = time.time()

        try:
            import pyautogui
            import pygetwindow as gw
        except ImportError:
            logger.warning("pyautogui 和 pygetwindow 未安装，使用备用方案")
            return self._fallback_screen_capture()

        while self._running:
            current_time = time.time()
            if current_time - last_time >= self.frame_interval:
                try:
                    if self.window_name:
                        # 截取指定窗口
                        windows = gw.getWindowsWithTitle(self.window_name)
                        if windows:
                            window = windows[0]
                            screenshot = pyautogui.screenshot(
                                region=(
                                    window.left,
                                    window.top,
                                    window.width,
                                    window.height,
                                )
                            )
                        else:
                            logger.warning(f"未找到窗口: {self.window_name}")
                            screenshot = pyautogui.screenshot(region=self.region)
                    else:
                        # 截取整个屏幕或指定区域
                        screenshot = pyautogui.screenshot(region=self.region)

                    # 转换为OpenCV格式
                    frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                    yield frame
                    last_time = current_time

                except Exception as e:
                    logger.error(f"截图失败: {e}")
                    yield np.zeros((480, 640, 3), dtype=np.uint8)

            time.sleep(0.001)  # 避免CPU占用过高

    def _fallback_screen_capture(self) -> Iterator[np.ndarray]:
        """
        备用屏幕截取方案
        """
        while self._running:
            # 返回黑色图像作为占位符
            yield np.zeros((480, 640, 3), dtype=np.uint8)
            time.sleep(self.frame_interval)

    def stop_capture(self):
        """
        停止截图
        """
        self._running = False

    def get_source_info(self) -> dict:
        return {
            "type": "window_capture",
            "window_name": self.window_name,
            "region": self.region,
            "fps": self.fps,
        }

    def is_stream(self) -> bool:
        return True


class ImageFileInputs(Inputs):
    def __init__(self, file_path: str):
        """
        初始化图片文件输入

        :param file_path: 图片文件路径
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {file_path}")

    def get_input_stream(self) -> Union[np.ndarray, str]:
        """
        获取图片（兼容ultralytics）
        """
        # 直接返回文件路径，ultralytics可以直接处理
        return str(self.file_path)

    def get_input_image(self) -> np.ndarray:
        """
        获取图片数组
        """
        image = cv2.imread(str(self.file_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"无法读取图片: {self.file_path}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return image

    def get_source_info(self) -> dict:
        return {
            "type": "image_file",
            "file_path": str(self.file_path),
            "exists": self.file_path.exists(),
            "size": self.file_path.stat().st_size if self.file_path.exists() else 0,
        }


class VideoFileInputs(Inputs):
    def __init__(
        self, file_path: str, start_frame: int = 0, max_frames: Optional[int] = None
    ):
        """
        初始化视频文件输入

        :param file_path: 视频文件路径
        :param start_frame: 起始帧
        :param max_frames: 最大帧数，None表示读取全部
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {file_path}")

        self.start_frame = start_frame
        self.max_frames = max_frames
        self._cap = None

    def get_input_stream(self) -> Union[Iterator[np.ndarray], str]:
        """
        获取视频流（兼容ultralytics）
        """
        return self.get_frame_generator()

    def get_frame_generator(self) -> Iterator[np.ndarray]:
        """
        获取视频帧生成器
        """
        cap = cv2.VideoCapture(str(self.file_path))
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {self.file_path}")
            return

        try:
            # 跳转到起始帧
            if self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                yield frame
                frame_count += 1

                # 检查最大帧数限制
                if self.max_frames and frame_count >= self.max_frames:
                    break

        finally:
            cap.release()

    def get_video_info(self) -> dict:
        """
        获取视频信息
        """
        cap = cv2.VideoCapture(str(self.file_path))
        if not cap.isOpened():
            return {}

        info = {
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            / cap.get(cv2.CAP_PROP_FPS),
        }
        cap.release()
        return info

    def get_source_info(self) -> dict:
        info = self.get_video_info()
        info.update(
            {
                "type": "video_file",
                "file_path": str(self.file_path),
                "start_frame": self.start_frame,
                "max_frames": self.max_frames,
            }
        )
        return info

    def is_stream(self) -> bool:
        return True


class WebcamInputs(Inputs):
    def __init__(self, camera_id: int = 0, fps: int = 30, resolution: tuple = None):
        """
        初始化摄像头输入

        :param camera_id: 摄像头ID
        :param fps: 帧率
        :param resolution: 分辨率 (width, height)
        """
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self._cap = None

    def get_input_stream(self) -> Union[Iterator[np.ndarray], int]:
        """
        获取摄像头流（兼容ultralytics）
        """
        # YOLO可以直接使用摄像头ID
        return self.camera_id

    def get_frame_generator(self) -> Iterator[np.ndarray]:
        """
        获取摄像头帧生成器
        """
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error(f"无法打开摄像头: {self.camera_id}")
            return

        try:
            # 设置分辨率
            if self.resolution:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

            # 设置帧率
            cap.set(cv2.CAP_PROP_FPS, self.fps)

            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("摄像头读取失败")
                    break
                yield frame

        finally:
            cap.release()

    def get_source_info(self) -> dict:
        return {
            "type": "webcam",
            "camera_id": self.camera_id,
            "fps": self.fps,
            "resolution": self.resolution,
        }

    def is_stream(self) -> bool:
        return True


class InputFactory:
    """
    输入源工厂类
    """

    @staticmethod
    def create_input(source: Union[str, int], **kwargs) -> Inputs:
        """
        根据源创建输入对象

        :param source: 输入源（文件路径、摄像头ID或特殊标识）
        :param kwargs: 额外参数
        """
        if isinstance(source, int):
            # 摄像头输入
            return WebcamInputs(source, **kwargs)

        elif isinstance(source, str):
            if source.startswith('window:'):
                # 窗口截图
                window_name = source[7:] if len(source) > 7 else None
                return WindowCaptureStreamInputs(window_name, **kwargs)

            elif source.startswith('screen'):
                # 屏幕截图
                return WindowCaptureStreamInputs(None, **kwargs)

            elif Path(source).exists():
                file_path = Path(source)
                if file_path.suffix.lower() in [
                    '.jpg',
                    '.jpeg',
                    '.png',
                    '.bmp',
                    '.tiff',
                ]:
                    # 图片文件
                    return ImageFileInputs(source)
                elif file_path.suffix.lower() in [
                    '.mp4',
                    '.avi',
                    '.mov',
                    '.mkv',
                    '.flv',
                ]:
                    # 视频文件
                    return VideoFileInputs(source, **kwargs)

            else:
                raise ValueError(f"无法识别的输入源: {source}")

        else:
            raise ValueError(f"不支持的输入源类型: {type(source)}")


# 使用示例
if __name__ == "__main__":
    # 创建不同类型的输入源
    factory = InputFactory()

    # 图片输入
    try:
        img_input = factory.create_input("tests/results.png")
        print(img_input.get_source_info())
    except FileNotFoundError:
        print("测试图片文件不存在")

    # # 摄像头输入
    # webcam_input = factory.create_input(0, fps=30)
    # print(webcam_input.get_source_info())

    # 窗口截图
    window_input = factory.create_input("window:Notepad", fps=10)
    print(window_input.get_source_info())
    cv2.imshow("Window Capture", next(window_input.get_input_stream()))
    cv2.waitKey(0)
