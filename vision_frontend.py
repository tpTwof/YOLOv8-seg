"""
视觉前端 — 给控制模块输出可抓目标信息
============================================
每帧输出一个 JSON 到 stdout，控制模块从 stdout 读取即可。

管线: RealSense RGB+Depth → YOLO-seg → 选最高置信度目标 → mask 中心 → 深度中位数 → 反投影 3D

用法:
    python vision_frontend.py                    # stdout JSON，无可视化
    python vision_frontend.py --show             # 同时显示 OpenCV 窗口
    python vision_frontend.py --model best.pt    # 指定模型
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "runs/segment/runs/segment/train/weights/best.pt",
)

RS_COLOR_WIDTH = 640
RS_COLOR_HEIGHT = 480
RS_FPS = 15

FRAME_NAME = "camera_color_optical_frame"

# ──────────────────────────────────────────────
# 手眼标定结果 (EYE_OUT_HAND): Camera → Base
# ──────────────────────────────────────────────
T_CAM2BASE = np.array([
    [0.10088699, -0.69368495, 0.71317811, 0.19659611],
    [0.99480199, 0.06038281, -0.08199345, -0.03518756],
    [0.01381392, 0.71774307, 0.69617100, -0.06704601],
    [0.0,        0.0,        0.0,         1.0       ],
])

# 可视化
MASK_ALPHA = 0.4
BOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2
TEXT_OFFSET_Y = -10
WINDOW_NAME = "Vision Frontend"


# ──────────────────────────────────────────────
# 可视化工具 (仅 --show 模式使用)
# ──────────────────────────────────────────────
def get_color(class_id: int) -> tuple:
    rng = np.random.RandomState(class_id + 1)
    return tuple(int(c) for c in rng.randint(50, 255, 3))


def draw_results(frame: np.ndarray, result, target_info: dict) -> np.ndarray:
    """在帧上画检测结果和选中目标的 3D 坐标。"""
    annotated = frame.copy()

    if result.masks is None:
        return annotated

    for i, mask in enumerate(result.masks.data):
        cls_id = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])
        label = f"{result.names[cls_id]} {conf:.2f}"
        color = get_color(cls_id)

        mask_np = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))
        mask_bool = mask_resized > 0.5

        for c in range(3):
            annotated[:, :, c] = np.where(
                mask_bool,
                (1 - MASK_ALPHA) * annotated[:, :, c] + MASK_ALPHA * color[c],
                annotated[:, :, c],
            )

        xyxy = result.boxes.xyxy[i].cpu().numpy().astype(int)
        cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, BOX_THICKNESS)
        text_y = max(xyxy[1] + TEXT_OFFSET_Y, 15)
        cv2.putText(annotated, label, (xyxy[0], text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

    # 画选中目标的 3D 坐标
    if target_info.get("has_target") and target_info.get("target"):
        t = target_info["target"]
        cx, cy = t["center_pixel"]
        cv2.circle(annotated, (int(cx), int(cy)), 6, (0, 0, 255), -1)
        pos = t["position_camera"]
        pos_base = t.get("position_base")
        if pos:
            txt_cam = f"Cam:({pos['x']:.3f},{pos['y']:.3f},{pos['z']:.3f})m"
            cv2.putText(annotated, txt_cam, (int(cx) + 10, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if pos_base:
            txt_base = f"Base:({pos_base['x']:.3f},{pos_base['y']:.3f},{pos_base['z']:.3f})m"
            cv2.putText(annotated, txt_base, (int(cx) + 10, int(cy) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return annotated


# ──────────────────────────────────────────────
# 目标信息提取
# ──────────────────────────────────────────────
def extract_mask_center(mask_binary: np.ndarray) -> tuple:
    """
    从二值 mask 计算质心 (u, v)。
    返回 (u, v) 像素坐标；mask 为空时返回 None。
    """
    ys, xs = np.where(mask_binary > 0)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def extract_depth_at_mask(depth_frame: np.ndarray, mask_binary: np.ndarray) -> tuple:
    """
    取 mask 区域内的深度中位数 (米) 和有效性。

    Args:
        depth_frame: 原始深度图 (uint16, 单位 mm)
        mask_binary: 二值 mask (uint8, 0/255)

    Returns:
        (depth_m, valid_depth)
    """
    mask_pixels = np.count_nonzero(mask_binary)
    if mask_pixels == 0:
        return 0.0, False

    depth_values = depth_frame[mask_binary > 0]
    valid_values = depth_values[depth_values > 0]

    if len(valid_values) < max(10, mask_pixels * 0.1):
        return 0.0, False

    median_mm = float(np.median(valid_values))
    return median_mm / 1000.0, True


def compute_grasp_yaw(mask_binary: np.ndarray) -> float:
    """
    用最小外接矩形计算 mask 的主方向角 (弧度)。
    返回 yaw 角；mask 太小时返回 0.0。
    """
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return 0.0
    rect = cv2.minAreaRect(largest)
    angle = rect[2]  # OpenCV 返回的角度是相对于水平线
    # 转为弧度，规范化到 [-pi/2, pi/2]
    yaw = np.deg2rad(angle)
    if yaw > np.pi / 2:
        yaw -= np.pi
    elif yaw < -np.pi / 2:
        yaw += np.pi
    return float(yaw)


def build_target_info(
    result,
    depth_frame: np.ndarray,
    intrinsics,
    frame_id: int,
) -> dict:
    """
    从一帧 YOLO 结果 + 深度图构建控制模块所需的目标信息。

    Args:
        result:      ultralytics 推理结果
        depth_frame: 对齐后的深度图 (uint16, mm)
        intrinsics:  RealSense 深度流内参
        frame_id:    帧编号

    Returns:
        完整的目标信息 dict
    """
    no_target = {
        "has_target": False,
        "target": None,
        "reason": "no_detection",
        "frame": FRAME_NAME,
        "timestamp": time.time(),
        "frame_id": frame_id,
    }

    if result.masks is None or len(result.masks) == 0:
        return no_target

    # 选置信度最高的目标
    confs = result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))

    cls_id = int(result.boxes.cls[best_idx])
    conf = float(confs[best_idx])
    class_name = result.names[cls_id]
    xyxy = result.boxes.xyxy[best_idx].cpu().numpy().tolist()

    # mask 二值图 (缩放到原图尺寸)
    mask_tensor = result.masks.data[best_idx]
    mask_np = mask_tensor.cpu().numpy()
    mask_resized = cv2.resize(mask_np, (depth_frame.shape[1], depth_frame.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

    # mask 面积 (像素)
    mask_area = float(np.count_nonzero(mask_binary))

    # mask 质心
    center = extract_mask_center(mask_binary)
    if center is None:
        return {**no_target, "reason": "mask_empty"}
    cu, cv_ = center

    # 深度
    depth_m, valid_depth = extract_depth_at_mask(depth_frame, mask_binary)

    # 3D 反投影
    position_camera = None
    if valid_depth and intrinsics is not None:
        import pyrealsense2 as rs
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [cu, cv_], depth_m)
        position_camera = {
            "x": round(point[0], 6),
            "y": round(point[1], 6),
            "z": round(point[2], 6),
        }

    # 抓取角
    grasp_yaw = compute_grasp_yaw(mask_binary)

    # 相机坐标 → 基座坐标
    position_base = None
    if position_camera is not None:
        p_cam = np.array([position_camera["x"],
                          position_camera["y"],
                          position_camera["z"],
                          1.0])
        p_base = T_CAM2BASE @ p_cam
        position_base = {
            "x": round(float(p_base[0]), 6),
            "y": round(float(p_base[1]), 6),
            "z": round(float(p_base[2]), 6),
        }

    return {
        "has_target": True,
        "target": {
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(conf, 4),
            "bbox_xyxy": [round(v, 1) for v in xyxy],
            "center_pixel": [round(cu, 1), round(cv_, 1)],
            "mask_area": round(mask_area, 1),
            "depth_m": round(depth_m, 4),
            "position_camera": position_camera,
            "position_base": position_base,
            "grasp_yaw": round(grasp_yaw, 4),
            "frame": FRAME_NAME,
            "timestamp": time.time(),
            "frame_id": frame_id,
            "valid_depth": valid_depth,
        },
    }


# ──────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────
def run(model_path: str, show: bool = False):
    import pyrealsense2 as rs

    # 加载模型
    print(f"[INFO] 加载模型: {model_path}", file=sys.stderr)
    model = YOLO(model_path)

    # RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RS_COLOR_WIDTH, RS_COLOR_HEIGHT, rs.format.bgr8, RS_FPS)
    config.enable_stream(rs.stream.depth, RS_COLOR_WIDTH, RS_COLOR_HEIGHT, rs.format.z16, RS_FPS)

    print("[INFO] 启动 RealSense...", file=sys.stderr)
    profile = pipeline.start(config)

    # 获取深度流内参 (用于反投影)
    depth_stream = profile.get_stream(rs.stream.depth)
    intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
    print(f"[INFO] 内参: fx={intrinsics.fx:.1f} fy={intrinsics.fy:.1f} "
          f"ppx={intrinsics.ppx:.1f} ppy={intrinsics.ppy:.1f}", file=sys.stderr)

    # 对齐: depth → color
    align = rs.align(rs.stream.color)

    # 预热
    print("[INFO] 预热中...", file=sys.stderr)
    for _ in range(30):
        try:
            pipeline.wait_for_frames(timeout_ms=10000)
        except RuntimeError:
            pass

    print("[INFO] 视觉前端已启动，JSON 输出到 stdout", file=sys.stderr)
    if show:
        print("[INFO] 按 'q' 退出可视化窗口", file=sys.stderr)

    frame_id = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError as e:
                print(f"[WARN] 取帧超时: {e}", file=sys.stderr)
                continue

            # 对齐深度到彩色
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame:
                continue

            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            # YOLO 推理
            results = model.predict(color_np, verbose=False)
            result = results[0]

            # 构建目标信息
            target_info = build_target_info(result, depth_np, intrinsics, frame_id)

            # 输出 JSON 到 stdout
            print(json.dumps(target_info, ensure_ascii=False), flush=True)

            # 可视化
            if show:
                annotated = draw_results(color_np, result, target_info)
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1

    finally:
        pipeline.stop()
        if show:
            cv2.destroyAllWindows()
        print(f"[INFO] 视觉前端停止，共处理 {frame_id} 帧", file=sys.stderr)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视觉前端 — 输出可抓目标 JSON")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help=f"YOLO 模型路径 (默认: {MODEL_PATH})")
    parser.add_argument("--show", action="store_true",
                        help="同时显示 OpenCV 可视化窗口")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.model, args.show)
