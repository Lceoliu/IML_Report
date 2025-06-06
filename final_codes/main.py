from pathlib import Path
import re
import cv2
import numpy as np

from typing import Union, Literal
from datetime import datetime
from ultralytics.utils.instance import Bboxes

from Inputs import (
    Inputs,
    WindowCaptureStreamInputs,
    ImageFileInputs,
    VideoFileInputs,
    InputFactory,
)

EXECUTE_TIME = datetime.now().strftime('%Y%m%d_%H%M%S')
TIME_IT_CACHE = []


def time_it_func(func):
    """
    装饰器，用于计算函数执行时间
    """

    def wrapper(*args, **kwargs):
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        save_path = Path(f"./analyze/analysis_{EXECUTE_TIME}.json").absolute()
        analysis_data = {
            "function": func.__name__,
            "execution_time": end_time - start_time,
            "timestamp": datetime.now().isoformat(),
        }
        if len(TIME_IT_CACHE) > 100:
            # 每100次调用保存一次
            with open(save_path, 'w') as f:
                import json

                json.dump(TIME_IT_CACHE, f, indent=4)
            TIME_IT_CACHE.clear()
        else:
            # 临时保存到内存中
            TIME_IT_CACHE.append(analysis_data)
        return result

    return wrapper


class Models:
    def __init__(
        self,
        model_path: str = "checkpoints/yolo_model.pt",
        model_type: Literal["yolo", "rtdetr", "fastsam"] = "yolo",
        image_provider: Inputs = None,
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.image_provider = image_provider
        self.is_running = True
        self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        加载模型
        """
        if self.model_type == "yolo":
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError(
                    "Please install the ultralytics package to use YOLO models."
                )
            self.model = YOLO(model_path)
        elif self.model_type == "rtdetr":
            try:
                from ultralytics import RTDETR
            except ImportError:
                raise ImportError(
                    "Please install the ultralytics package to use RTDETR models."
                )
            self.model = RTDETR(model_path)
        elif self.model_type == "fastsam":
            try:
                from ultralytics import FastSAM
            except ImportError:
                raise ImportError(
                    "Please install the ultralytics package to use FastSAM models."
                )
            self.model = FastSAM(model_path)
        else:
            raise ValueError(
                f"Unsupported model type: {self.model_type}. Supported types are 'yolo', 'rtdetr', and 'fastsam'."
            )
        print(f"Model loaded from {model_path} with type {self.model_type}")

    def prune_model(self, prune_percent: float = 0.1):
        """
        对模型进行剪枝
        """
        if self.model is None:
            print("No model loaded to prune")
            return

        try:
            import torch
            import torch.nn.utils.prune as prune
        except ImportError:
            print("PyTorch is required for model pruning")
            return

        print(f"Starting model pruning with {prune_percent*100}% pruning rate...")

        # 获取模型的所有参数
        parameters_to_prune = []
        for name, module in self.model.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))

        if not parameters_to_prune:
            print("No suitable layers found for pruning")
            return

        # 使用全局非结构化剪枝
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_percent,
        )

        # 永久移除被剪枝的权重
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        print(f"Model pruning completed. Pruned {prune_percent*100}% of parameters.")

    @time_it_func
    def predict_from_image(self, image: np.ndarray, resize: bool = True):
        """
        进行预测
        """
        if self.model is not None:
            if resize:
                image = cv2.resize(image, (640, 640))
            results = self.model.predict(image)
            return results
        return None

    def release_model(self):
        """
        释放模型
        """
        if self.model is not None:
            del self.model
            self.model = None
            self.is_running = False
            print("Model released.")

    def get_image(self):
        """
        获取图像
        """
        if self.image_provider is None:
            return None

        return self.image_provider.get_input_stream()

    def draw_results_on_image(self, image: np.ndarray, results):
        """
        在图像上绘制预测结果
        """
        if not results or len(results) == 0:
            return image

        # 获取第一个结果（通常只有一个）
        result = results[0]
        orig_size = result.orig_shape if hasattr(result, 'orig_shape') else None
        scaling = (
            (
                result.orig_shape[0] / image.shape[0],
                result.orig_shape[1] / image.shape[1],
            )
            if orig_size
            else (1.0, 1.0)
        )

        # 创建图像副本用于绘制
        annotated_image = image.copy()

        # 检查是否有检测结果
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            for i, box in enumerate(boxes):
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # 应用缩放
                x1 = int(x1 / scaling[1])
                y1 = int(y1 / scaling[0])
                x2 = int(x2 / scaling[1])
                y2 = int(y2 / scaling[0])

                # 获取置信度
                conf = float(box.conf[0]) if box.conf is not None else 0.0

                # 获取类别
                if box.cls is not None:
                    cls_id = int(box.cls[0])
                    # 获取类别名称（如果模型有names属性）
                    if hasattr(result, 'names') and result.names:
                        cls_name = result.names.get(cls_id, f'Class_{cls_id}')
                    else:
                        cls_name = f'Class_{cls_id}'
                else:
                    cls_name = 'Unknown'

                # 选择颜色（根据类别ID）
                color = self._get_color(cls_id if box.cls is not None else 0)

                # 绘制边界框
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)

                # 准备标签文本
                label = f'{cls_name}: {conf:.2f}'

                # 计算文本大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # 绘制标签背景
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - text_height - baseline),
                    (x1 + text_width, y1),
                    color,
                    -1,
                )

                # 绘制标签文本
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # 如果是分割模型，处理mask
        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            for mask in masks:
                # 将mask应用到图像上
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_colored = np.zeros_like(image)
                mask_colored[mask_resized > 0.5] = [0, 255, 0]  # 绿色mask
                annotated_image = cv2.addWeighted(
                    annotated_image, 0.8, mask_colored, 0.2, 0
                )

        return annotated_image

    def _get_color(self, class_id: int):
        """
        根据类别ID获取颜色
        """
        colors = [
            (255, 0, 0),  # 红色
            (0, 255, 0),  # 绿色
            (0, 0, 255),  # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 青色
            (128, 0, 128),  # 紫罗兰
            (255, 165, 0),  # 橙色
            (255, 192, 203),  # 粉色
            (128, 128, 128),  # 灰色
        ]
        return colors[class_id % len(colors)]

    def show_prediction(self, save_video: bool = False):
        """
        在新窗口中显示预测结果
        """
        if self.image_provider is None:
            print("No image provider available")
            return None

        # avg_fps = 0
        window_name = "AI Model Prediction"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            if not self.image_provider.is_stream():
                # 单张图片处理
                if hasattr(self.image_provider, 'get_input_image'):
                    image = self.image_provider.get_input_image()
                else:
                    # 如果没有get_input_image方法，尝试从get_input_stream获取
                    stream = self.image_provider.get_input_stream()
                    if isinstance(stream, str):  # 文件路径
                        image = cv2.imread(stream)
                    elif isinstance(stream, np.ndarray):  # 图像数组
                        image = stream
                    else:
                        print("Cannot get image from provider")
                        return

                if image is not None:
                    print("Processing single image...")
                    results = self.predict_from_image(image)
                    if results:
                        annotated_image = self.draw_results_on_image(image, results)
                        cv2.imshow(window_name, annotated_image)
                        print("Press any key to close the window")
                        cv2.waitKey(0)
                    else:
                        cv2.imshow(window_name, image)
                        cv2.waitKey(0)
                else:
                    print("Failed to load image")
            else:
                # 流式输入处理
                print("Starting stream processing... Press 'q' to quit")
                stream_gen = self.image_provider.get_input_stream()

                if hasattr(self.image_provider, 'get_frame_generator'):
                    # 使用frame generator
                    stream_gen = self.image_provider.get_frame_generator()

                video_writer = None
                # 如果需要保存视频
                if save_video:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                    info = self.image_provider.get_source_info()
                    output_file = Path(info.get('file_path', 'output.mp4'))
                    output_file = output_file.parent / f"{output_file.stem}_output.mp4"
                    print(f"Saving video to {output_file}")
                    size = (info.get('width', 640), info.get('height', 480))
                    fps = info.get('fps', 30)
                    video_writer = cv2.VideoWriter(
                        output_file.resolve(), fourcc, fps, size
                    )

                frame_count = 0
                fps_counter = cv2.getTickCount()

                for image in stream_gen:
                    if not self.is_running:
                        break

                    if image is not None:
                        frame_count += 1

                        # 进行预测
                        results = self.predict_from_image(image)

                        if results:
                            # 绘制预测结果
                            annotated_image = self.draw_results_on_image(image, results)

                            # 添加FPS信息
                            if frame_count % 30 == 0:  # 每30帧计算一次FPS
                                current_time = cv2.getTickCount()
                                fps = 30 / (
                                    (current_time - fps_counter)
                                    / cv2.getTickFrequency()
                                )
                                fps_counter = current_time

                            # 在图像上显示帧数和FPS信息
                            info_text = f"Frame: {frame_count}  FPS: {fps:.2f}"
                            cv2.putText(
                                annotated_image,
                                info_text,
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                            )
                        else:
                            annotated_image = image

                        # 显示图像
                        cv2.imshow(window_name, annotated_image)
                        if video_writer is not None:
                            video_writer.write(annotated_image)

                        # 检查退出条件
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' 或 ESC 键
                            print("Stopping stream...")
                            break

                    # 如果是窗口截图类型，需要控制循环
                    if hasattr(self.image_provider, 'stop_capture'):
                        if not self.is_running:
                            self.image_provider.stop_capture()

                            break

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error during prediction: {e}")
        finally:
            cv2.destroyAllWindows()
            if video_writer is not None:
                video_writer.release()
            if hasattr(self.image_provider, 'stop_capture'):
                self.image_provider.stop_capture()


def main(args):
    input_method = args.input_method
    if input_method == "window_capture":
        inputs = WindowCaptureStreamInputs(
            window_name=args.window_name,
            fps=args.fps,
        )
    elif input_method == "image_file":
        inputs = ImageFileInputs(
            image_path=args.image_path,
        )
    elif input_method == "video_file":
        inputs = VideoFileInputs(
            file_path=args.video_path,
        )
    else:
        raise ValueError(
            f"Unsupported input method: {input_method}. Supported methods are 'window_capture', 'image_file', and 'video_file'."
        )

    print(f"Using input method: {input_method}\n info: {inputs.get_source_info()}")
    model = Models(
        model_path=args.model_path, model_type=args.model_type, image_provider=inputs
    )
    if args.prune_percent > 0:
        model.prune_model(prune_percent=args.prune_percent)

    model.show_prediction(save_video=args.save_output_video)
    model.release_model()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run object detection models.")
    parser.add_argument(
        "--input_method",
        type=str,
        choices=["window_capture", "image_file", "video_file"],
        required=True,
        help="Input method for the model.",
    )
    parser.add_argument(
        "--window_name",
        type=str,
        help="Name of the window to capture. e.g., 'window:Google Chrome'",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for capture."
    )
    parser.add_argument("--image_path", type=str, help="Path to the image file.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/yolo_model.pt",
        help="Path to the model file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["yolo", "rtdetr", "fastsam"],
        default="yolo",
        help="Type of the model to use.",
    )
    parser.add_argument(
        "--prune_percent",
        type=float,
        default=0,
        help="Pruning percent for the model.",
    )
    parser.add_argument(
        "--save_output_video",
        action="store_true",
        default=True,
        help="Save the output video with predictions.",
    )
    args = parser.parse_args()
    main(args)
    if TIME_IT_CACHE and len(TIME_IT_CACHE) > 0:
        with open(Path(f"./analyze/analysis_{EXECUTE_TIME}.json").absolute(), 'w') as f:
            import json

            json.dump(TIME_IT_CACHE, f, indent=4)
