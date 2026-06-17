"""
YOLO → LabelMe 格式转换脚本
==============================
将 YOLO segmentation 标签 (.txt) 转换为 LabelMe JSON 格式 (.json)，
以便使用 LabelMe 工具进行人工微调标注。

YOLO segmentation 格式:
    class_id x1 y1 x2 y2 ... xn yn  (归一化坐标 0~1)

LabelMe JSON 格式:
    {
        "version": "5.5.0",
        "flags": {},
        "shapes": [{"label": "cube", "points": [[x,y],...], "shape_type": "polygon", ...}],
        "imagePath": "xxx.jpg",
        "imageData": null,
        "imageHeight": 480,
        "imageWidth": 640
    }

用法:
    python scripts/yolo_to_labelme.py
    python scripts/yolo_to_labelme.py --image-dir dataset/images --label-dir dataset/labels --output-dir dataset/labelme

输出:
    dataset/labelme/train/*.json  -- 训练集 LabelMe 标注
    dataset/labelme/val/*.json    -- 验证集 LabelMe 标注
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2


# ──────────────────────────────────────────────
# 核心函数
# ──────────────────────────────────────────────

def yolo_txt_to_labelme_json(
    txt_path: str,
    image_path: str,
    json_path: str,
    class_names: Optional[Dict[int, str]] = None,
) -> int:
    """
    将单个 YOLO 标签文件转换为 LabelMe JSON 格式。

    Args:
        txt_path:     YOLO .txt 标签文件路径
        image_path:   对应的图片路径 (用于读取尺寸)
        json_path:    输出 JSON 文件路径
        class_names:  类别ID到名称的映射字典，例如 {0: "cube"}

    Returns:
        转换的目标数量
    """
    # 默认类别名
    if class_names is None:
        class_names = {0: "cube"}

    # 读取图片尺寸
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] 无法读取图片: {image_path}")
        return 0
    img_h, img_w = img.shape[:2]

    # 读取 YOLO 标签
    shapes = []
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  # 至少 class_id + 3个点(6个坐标)
                    continue

                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                # 将归一化坐标转为像素坐标
                points = []
                for i in range(0, len(coords), 2):
                    x_pixel = coords[i] * img_w
                    y_pixel = coords[i + 1] * img_h
                    points.append([round(x_pixel, 2), round(y_pixel, 2)])

                label = class_names.get(cls_id, str(cls_id))

                shapes.append({
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None,
                })

    # 构建 LabelMe JSON
    labelme_data = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,  # 不内嵌图片数据，LabelMe 会自动从 imagePath 加载
        "imageHeight": img_h,
        "imageWidth": img_w,
    }

    # 写入 JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    return len(shapes)


def convert_directory(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    class_names: Optional[Dict[int, str]] = None,
) -> Tuple[int, int]:
    """
    将一个目录中所有 YOLO 标签转换为 LabelMe JSON。

    Args:
        image_dir:    图片目录
        label_dir:    YOLO 标签目录
        output_dir:   LabelMe JSON 输出目录
        class_names:  类别ID到名称的映射

    Returns:
        (处理文件数, 转换目标总数)
    """
    os.makedirs(output_dir, exist_ok=True)

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
        stem = Path(img_file).stem
        img_path = os.path.join(image_dir, img_file)
        txt_path = os.path.join(label_dir, stem + ".txt")
        json_path = os.path.join(output_dir, stem + ".json")

        n = yolo_txt_to_labelme_json(txt_path, img_path, json_path, class_names)
        total_objects += n

        # 创建图片软链接到 labelme 目录，使 LabelMe 工具能直接打开
        link_path = os.path.join(output_dir, img_file)
        if not os.path.exists(link_path):
            os.symlink(os.path.abspath(img_path), link_path)

    return len(image_files), total_objects


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(
    image_dir: str = "dataset/images",
    label_dir: str = "dataset/labels",
    output_dir: str = "dataset/labelme",
    class_names: Optional[Dict[int, str]] = None,
) -> Tuple[int, int]:
    """
    执行完整的 YOLO → LabelMe 转换流程。

    Args:
        image_dir:   图片根目录 (下含 train/ 和 val/)
        label_dir:   YOLO 标签根目录 (下含 train/ 和 val/)
        output_dir:  LabelMe JSON 输出根目录 (下含 train/ 和 val/)
        class_names: 类别ID到名称的映射

    Returns:
        (处理文件总数, 转换目标总数)
    """
    total_imgs, total_objs = 0, 0

    for split in ["train", "val"]:
        img_split_dir = os.path.join(image_dir, split)
        lbl_split_dir = os.path.join(label_dir, split)
        out_split_dir = os.path.join(output_dir, split)

        if not os.path.isdir(img_split_dir):
            print(f"  [SKIP] 目录不存在: {img_split_dir}")
            continue

        print(f"[INFO] 正在转换 {split} 集...")
        n_imgs, n_objs = convert_directory(img_split_dir, lbl_split_dir, out_split_dir, class_names)
        total_imgs += n_imgs
        total_objs += n_objs
        print(f"  [OK] {split}: {n_imgs} 个文件, {n_objs} 个标注")

    print(f"\n[INFO] 转换完成: 共 {total_imgs} 个文件, {total_objs} 个标注")
    return total_imgs, total_objs


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO segmentation 标签 → LabelMe JSON 转换")
    parser.add_argument("--image-dir", type=str, default="dataset/images",
                        help="图片根目录 (默认: dataset/images)")
    parser.add_argument("--label-dir", type=str, default="dataset/labels",
                        help="YOLO 标签根目录 (默认: dataset/labels)")
    parser.add_argument("--output-dir", type=str, default="dataset/labelme",
                        help="LabelMe JSON 输出目录 (默认: dataset/labelme)")
    parser.add_argument("--classes", type=str, default="0:cube",
                        help="类别映射，格式: 'id:name,id:name' (默认: '0:cube')")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # 解析类别映射
    class_names = {}
    for pair in args.classes.split(","):
        cid, cname = pair.strip().split(":")
        class_names[int(cid)] = cname

    run(args.image_dir, args.label_dir, args.output_dir, class_names)
