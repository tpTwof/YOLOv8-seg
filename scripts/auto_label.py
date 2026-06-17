"""
自动标注脚本
=============
使用已有的 YOLOv8-seg 模型对图片进行推理，提取 segmentation 多边形，
生成 YOLO segmentation 格式的标签文件。

YOLO segmentation 格式 (每行一个目标):
    class_id x1 y1 x2 y2 ... xn yn
    坐标为归一化的多边形顶点 (0~1)

用法:
    python scripts/auto_label.py
    python scripts/auto_label.py --image-dir dataset/images --model best.pt

输出:
    dataset/labels/train/*.txt  -- 训练集标签
    dataset/labels/val/*.txt    -- 验证集标签
"""

import argparse
import os
from pathlib import Path
from typing import Tuple

from ultralytics import YOLO


# ──────────────────────────────────────────────
# 核心函数
# ──────────────────────────────────────────────

def label_single_image(model: YOLO, image_path: str, label_path: str, conf_threshold: float = 0.25) -> int:
    """
    对单张图片进行推理，将检测结果保存为 YOLO segmentation 标签。

    Args:
        model:         已加载的 YOLO 模型
        image_path:    输入图片路径
        label_path:    输出标签路径 (.txt)
        conf_threshold: 置信度阈值，低于此值的检测结果将被忽略

    Returns:
        检测到的目标数量
    """
    results = model.predict(image_path, verbose=False, conf=conf_threshold)
    result = results[0]

    # 没有检测到目标，创建空文件
    if result.masks is None or len(result.masks) == 0:
        Path(label_path).touch()
        return 0

    img_h, img_w = result.orig_shape  # 原图尺寸
    lines = []

    for i, mask_polygon in enumerate(result.masks.xy):
        # mask.xy[i] 是多边形顶点的像素坐标 (Nx2)
        cls_id = int(result.boxes.cls[i])

        # 归一化到 0~1
        points = []
        for x, y in mask_polygon:
            points.append(f"{x / img_w:.6f}")
            points.append(f"{y / img_h:.6f}")

        line = f"{cls_id} " + " ".join(points)
        lines.append(line)

    # 写入标签文件
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" if lines else "")

    return len(lines)


def label_directory(model: YOLO, image_dir: str, label_dir: str, conf_threshold: float = 0.25) -> Tuple[int, int]:
    """
    对一个目录中的所有图片进行自动标注。

    Args:
        model:         已加载的 YOLO 模型
        image_dir:     图片目录
        label_dir:     标签输出目录
        conf_threshold: 置信度阈值

    Returns:
        (处理图片数, 检测目标总数)
    """
    os.makedirs(label_dir, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in image_extensions
    ])

    if not image_files:
        print(f"  [WARN] {image_dir} 中无图片")
        return 0, 0

    total_objects = 0
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        lbl_path = os.path.join(label_dir, Path(img_file).stem + ".txt")
        n = label_single_image(model, img_path, lbl_path, conf_threshold)
        total_objects += n

    return len(image_files), total_objects


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(image_dir: str = "dataset/images", model_path: str = "best.pt", conf_threshold: float = 0.25) -> Tuple[int, int]:
    """
    执行完整的自动标注流程。

    Args:
        image_dir:     图片根目录 (下含 train/ 和 val/)
        model_path:    YOLO 模型路径
        conf_threshold: 置信度阈值

    Returns:
        (处理图片总数, 检测目标总数)
    """
    label_dir = image_dir.replace("images", "labels")

    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    print(f"[INFO] 模型类别: {class_names}")
    print(f"[INFO] 置信度阈值: {conf_threshold}")

    total_imgs, total_objs = 0, 0

    for split in ["train", "val"]:
        img_split_dir = os.path.join(image_dir, split)
        lbl_split_dir = os.path.join(label_dir, split)

        if not os.path.isdir(img_split_dir):
            print(f"  [SKIP] 目录不存在: {img_split_dir}")
            continue

        print(f"\n[INFO] 正在标注 {split} 集...")
        n_imgs, n_objs = label_directory(model, img_split_dir, lbl_split_dir, conf_threshold)
        total_imgs += n_imgs
        total_objs += n_objs
        print(f"  [OK] {split}: {n_imgs} 张图片, {n_objs} 个目标")

    print(f"\n[INFO] 自动标注完成: 共 {total_imgs} 张图片, {total_objs} 个目标")
    return total_imgs, total_objs


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="使用 YOLOv8-seg 模型自动标注图片")
    parser.add_argument("--image-dir", type=str, default="dataset/images",
                        help="图片根目录 (默认: dataset/images)")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="YOLO 模型路径 (默认: best.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args.image_dir, args.model, args.conf)
