"""
YOLOv8-seg 实例分割推理脚本
支持四种输入模式: 图片 | 视频 | 实时摄像头(内置相机) | Intel RealSense 深度相机
输出: 带分割掩码、边界框、类别标签的可视化结果

用法:
    python infer.py --source image.jpg              # 图片推理
    python infer.py --source video.mp4              # 视频推理
    python infer.py --source camera                 # 内置摄像头
    python infer.py --source realsense              # RealSense 深度相机
    python infer.py --source realsense --depth      # RealSense + 深度图叠加
    python infer.py --source image.jpg --no-show    # 不弹窗，只保存
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils.logger import write_log

# ──────────────────────────────────────────────
# 常量配置
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs/segment/runs/segment/train/weights/best.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_CAM_INDEX = 0  # 内置相机索引
WINDOW_NAME = "YOLOv8-seg Inference"

# RealSense 默认分辨率
RS_COLOR_WIDTH = 640
RS_COLOR_HEIGHT = 480
RS_FPS = 15

# 可视化参数
MASK_ALPHA = 0.4       # 掩码透明度
BOX_THICKNESS = 2      # 边界框线宽
FONT_SCALE = 0.7       # 字体大小
FONT_THICKNESS = 2     # 字体粗细
TEXT_OFFSET_Y = -10    # 标签文字相对框顶部的偏移
DEPTH_ALPHA = 0.5      # 深度图叠加透明度


# ──────────────────────────────────────────────
# 可视化工具
# ──────────────────────────────────────────────
def get_color(class_id: int) -> tuple:
    """根据类别ID生成固定颜色(BGR)，保证同类目标颜色一致。"""
    rng = np.random.RandomState(class_id + 1)
    return tuple(int(c) for c in rng.randint(50, 255, 3))


def draw_results(frame: np.ndarray, result) -> np.ndarray:
    """
    在单帧上绘制分割掩码、边界框和类别标签。

    Args:
        frame:  原始图像 (BGR)
        result: ultralytics 推理结果对象

    Returns:
        绘制后的图像
    """
    annotated = frame.copy()

    if result.masks is None:
        return annotated

    for i, mask in enumerate(result.masks.data):
        cls_id = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])
        label = f"{result.names[cls_id]} {conf:.2f}"
        color = get_color(cls_id)

        # 掩码缩放到原图尺寸
        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
        mask_bool = mask_resized > 0.5

        # 叠加彩色掩码
        for c in range(3):
            annotated[:, :, c] = np.where(
                mask_bool,
                (1 - MASK_ALPHA) * annotated[:, :, c] + MASK_ALPHA * color[c],
                annotated[:, :, c],
            )

        # 边界框
        xyxy = result.boxes.xyxy[i].cpu().numpy().astype(int)
        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, BOX_THICKNESS)

        # 类别标签
        text_y = max(xyxy[1] + TEXT_OFFSET_Y, 15)
        cv2.putText(annotated, label, (xyxy[0], text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

    return annotated


def colorize_depth(depth_frame: np.ndarray) -> np.ndarray:
    """
    将深度图转为伪彩色(BGR)，便于可视化。

    Args:
        depth_frame: 原始深度数据 (单位: mm, uint16)

    Returns:
        伪彩色 BGR 图像
    """
    depth_clipped = np.clip(depth_frame, 0, 5000)  # 截断到 5m
    depth_normalized = (depth_clipped / 5000 * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)


def save_result(frame: np.ndarray, source_name: str, output_dir: str) -> str:
    """保存结果帧到输出目录，返回保存路径。"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{Path(source_name).stem}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath


# ──────────────────────────────────────────────
# 推理函数
# ──────────────────────────────────────────────
def infer_image(model: YOLO, source: str, show: bool = True) -> dict:
    """单张图片推理。"""
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {source}")

    results = model.predict(source, verbose=False)
    annotated = draw_results(frame, results[0])
    saved_path = save_result(annotated, os.path.basename(source), OUTPUT_DIR)

    if show:
        cv2.imshow(WINDOW_NAME, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    n_objects = len(results[0].boxes) if results[0].boxes is not None else 0
    return {"n_objects": n_objects, "saved_path": saved_path}


def infer_video(model: YOLO, source: str, show: bool = True) -> dict:
    """视频文件推理，逐帧处理并保存结果视频。"""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"{Path(source).stem}_{timestamp}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count, total_objects = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, verbose=False)
        annotated = draw_results(frame, results[0])
        writer.write(annotated)

        n_obj = len(results[0].boxes) if results[0].boxes is not None else 0
        total_objects += n_obj
        frame_count += 1

        if show:
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    return {"frame_count": frame_count, "total_objects": total_objects, "saved_path": out_path}


def infer_camera(model: YOLO, cam_index: int = DEFAULT_CAM_INDEX, show: bool = True) -> dict:
    """实时内置摄像头推理，按 q 退出。"""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头(索引 {cam_index})，请检查设备连接")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"camera_{timestamp}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))

    frame_count, total_objects = 0, 0
    print("[INFO] 实时检测已启动，按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取摄像头帧失败，退出")
            break

        results = model.predict(frame, verbose=False)
        annotated = draw_results(frame, results[0])
        writer.write(annotated)

        n_obj = len(results[0].boxes) if results[0].boxes is not None else 0
        total_objects += n_obj
        frame_count += 1

        if show:
            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    return {"frame_count": frame_count, "total_objects": total_objects, "saved_path": out_path}


def infer_realsense(model: YOLO, show: bool = True, show_depth: bool = False) -> dict:
    """
    Intel RealSense 深度相机推理，按 q 退出。
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise ImportError("请先安装 pyrealsense2: pip install pyrealsense2")

    pipeline = rs.pipeline()
    config = rs.config()

    # 建议 USB 2.0 下先用 15 FPS
    color_w = RS_COLOR_WIDTH
    color_h = RS_COLOR_HEIGHT
    fps = RS_FPS

    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)

    if show_depth:
        config.enable_stream(rs.stream.depth, color_w, color_h, rs.format.z16, fps)

    started = False
    writer = None

    try:
        print("[INFO] 正在启动 RealSense pipeline...")
        profile = pipeline.start(config)
        started = True

        # 打印 USB 类型，方便判断是不是 USB2
        device = profile.get_device()
        try:
            usb_type = device.get_info(rs.camera_info.usb_type_descriptor)
            print(f"[INFO] RealSense USB Type: {usb_type}")
        except Exception:
            pass

        # 预热，丢掉前几帧
        print("[INFO] RealSense 预热中...")
        for _ in range(30):
            try:
                pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError as e:
                print(f"[WARN] 预热取帧超时: {e}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_w = color_w * 2 if show_depth else color_w
        out_path = os.path.join(OUTPUT_DIR, f"realsense_{timestamp}.mp4")
        writer = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_w, color_h)
        )

        frame_count, total_objects = 0, 0
        print("[INFO] RealSense 检测已启动，按 'q' 退出")

        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError as e:
                print(f"[WARN] RealSense 等待帧超时: {e}")
                continue

            color_frame = frames.get_color_frame()
            if not color_frame:
                print("[WARN] 没有获取到 color frame")
                continue

            frame = np.asanyarray(color_frame.get_data())

            results = model.predict(frame, verbose=False)
            annotated = draw_results(frame, results[0])

            n_obj = len(results[0].boxes) if results[0].boxes is not None else 0
            total_objects += n_obj
            frame_count += 1

            if show_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_np = np.asanyarray(depth_frame.get_data())
                    depth_color = colorize_depth(depth_np)

                    if depth_color.shape[:2] != annotated.shape[:2]:
                        depth_color = cv2.resize(depth_color, (annotated.shape[1], annotated.shape[0]))

                    display = np.hstack([annotated, depth_color])
                else:
                    # writer 期望宽度是双倍，所以没有 depth 时补一张黑图
                    blank = np.zeros_like(annotated)
                    display = np.hstack([annotated, blank])
            else:
                display = annotated

            writer.write(display)

            if show:
                cv2.imshow(WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        if started:
            pipeline.stop()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    return {
        "frame_count": frame_count,
        "total_objects": total_objects,
        "saved_path": out_path
    }

# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
def detect_source_type(source: str) -> str:
    """
    自动判断输入源类型。

    Returns:
        "image" | "video" | "camera" | "realsense"
    """
    source_lower = source.lower()
    if source_lower == "camera":
        return "camera"
    if source_lower == "realsense":
        return "realsense"

    ext = Path(source).suffix.lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
        return "image"
    if ext in (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"):
        return "video"

    raise ValueError(f"不支持的文件格式: {ext}")


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="YOLOv8-seg 实例分割推理")
    parser.add_argument("--source", type=str, required=True,
                        help="输入源: 图片路径 | 视频路径 | 'camera' | 'realsense'")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help=f"模型路径 (默认: {MODEL_PATH})")
    parser.add_argument("--cam-index", type=int, default=DEFAULT_CAM_INDEX,
                        help="内置摄像头索引 (默认: 0)")
    parser.add_argument("--depth", action="store_true",
                        help="RealSense 模式下并排显示深度伪彩色图")
    parser.add_argument("--no-show", action="store_true",
                        help="不显示弹窗，仅保存结果")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 加载模型
    print(f"[INFO] 加载模型: {args.model}")
    model = YOLO(args.model)
    source_type = detect_source_type(args.source)
    print(f"[INFO] 输入类型: {source_type} | 输入源: {args.source}")

    show = not args.no_show

    # 根据输入类型分发推理
    dispatch = {
        "image":     lambda: infer_image(model, args.source, show),
        "video":     lambda: infer_video(model, args.source, show),
        "camera":    lambda: infer_camera(model, args.cam_index, show),
        "realsense": lambda: infer_realsense(model, show, args.depth),
    }
    result = dispatch[source_type]()

    # 构建日志
    log_lines = [
        "## 推理完成",
        f"- 输入类型: {source_type}",
        f"- 输入源: {args.source}",
        f"- 结果保存: {result.get('saved_path', 'N/A')}",
    ]
    if "n_objects" in result:
        log_lines.append(f"- 检测目标数: {result['n_objects']}")
    if "frame_count" in result:
        log_lines.append(f"- 处理帧数: {result['frame_count']}")
        log_lines.append(f"- 累计检测目标: {result['total_objects']}")
    log_lines += [
        "",
        "## 技术栈",
        "- 模型: YOLOv8-seg (实例分割)",
        "- 推理框架: ultralytics",
        "- 可视化: OpenCV (掩码叠加 + 边界框 + 标签)",
    ]
    if source_type == "realsense":
        log_lines.append("- 相机SDK: pyrealsense2 (Intel RealSense)")
        if args.depth:
            log_lines.append("- 深度可视化: 伪彩色热力图 (JET colormap)")

    log_path = write_log("\n".join(log_lines))
    print(f"[INFO] 日志已保存: {log_path}")
    print(f"[INFO] 结果已保存: {result.get('saved_path', 'N/A')}")


if __name__ == "__main__":
    main()
