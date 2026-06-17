"""
LabelMe → YOLO 格式转换脚本
==============================
将 LabelMe JSON 标注 (.json) 转换为 YOLO segmentation 标签 (.txt)，
用于将人工微调后的标注转回训练格式。

LabelMe JSON 格式:
    shapes[].label         → 类别名
    shapes[].points        → 多边形顶点 (像素坐标)
    shapes[].shape_type    → "polygon" 或 "rectangle"
    imageWidth/imageHeight → 图片尺寸 (用于归一化)

YOLO segmentation 格式 (每行一个目标):
    class_id x1 y1 x2 y2 ... xn yn  (归一化坐标 0~1)

用法:
    python scripts/labelme_to_yolo.py
    python scripts/labelme_to_yolo.py --labelme-dir dataset/labelme --output-dir dataset/labels

注意:
    - 会覆盖 output-dir 中已有的同名 .txt 文件
    - 支持 polygon 和 rectangle 两种 shape_type
    - rectangle 会被转为 4 个顶点的多边形
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


# ──────────────────────────────────────────────
# 核心函数
# ──────────────────────────────────────────────

def build_class_map(labelme_dir: str) -> Dict[str, int]:
    """
    扫描 LabelMe 目录，自动构建 label_name → class_id 映射。
    按字母序排列，保证每次运行结果一致。

    Args:
        labelme_dir: LabelMe JSON 所在目录

    Returns:
        {"cube": 0, "sphere": 1, ...}
    """
    labels = set()
    for f in os.listdir(labelme_dir):
        if not f.endswith(".json"):
            continue
        data = json.load(open(os.path.join(labelme_dir, f), encoding="utf-8"))
        for shape in data.get("shapes", []):
            labels.add(shape["label"])

    return {name: idx for idx, name in enumerate(sorted(labels))}


def labelme_json_to_yolo_txt(
    json_path: str,
    txt_path: str,
    class_map: Dict[str, int],
) -> int:
    """
    将单个 LabelMe JSON 转换为 YOLO segmentation 标签。

    Args:
        json_path:  LabelMe JSON 文件路径
        txt_path:   输出 YOLO .txt 文件路径
        class_map:  类别名到ID的映射 {"cube": 0}

    Returns:
        转换的目标数量
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth", 0)
    img_h = data.get("imageHeight", 0)
    if img_w == 0 or img_h == 0:
        print(f"  [WARN] 图片尺寸缺失: {json_path}")
        return 0

    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"]
        if label not in class_map:
            print(f"  [WARN] 未知类别 '{label}'，跳过 ({json_path})")
            continue

        cls_id = class_map[label]
        points = shape["points"]
        shape_type = shape.get("shape_type", "polygon")

        # rectangle → 转为 4 个顶点的多边形
        if shape_type == "rectangle" and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        # 像素坐标 → 归一化坐标
        coords = []
        for x, y in points:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")

        lines.append(f"{cls_id} " + " ".join(coords))

    # 写入文件
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" if lines else "")

    return len(lines)


def convert_directory(
    labelme_dir: str,
    output_dir: str,
    class_map: Dict[str, int],
) -> Tuple[int, int]:
    """
    将一个目录中所有 LabelMe JSON 转换为 YOLO 标签。

    Args:
        labelme_dir: LabelMe JSON 目录
        output_dir:  YOLO .txt 输出目录
        class_map:   类别名到ID的映射

    Returns:
        (处理文件数, 转换目标总数)
    """
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted([
        f for f in os.listdir(labelme_dir)
        if f.endswith(".json")
    ])

    if not json_files:
        print(f"  [WARN] {labelme_dir} 中无 JSON 文件")
        return 0, 0

    total_objects = 0
    for json_file in json_files:
        stem = Path(json_file).stem
        json_path = os.path.join(labelme_dir, json_file)
        txt_path = os.path.join(output_dir, stem + ".txt")

        n = labelme_json_to_yolo_txt(json_path, txt_path, class_map)
        total_objects += n

    return len(json_files), total_objects


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(
    labelme_dir: str = "dataset/labelme",
    output_dir: str = "dataset/labels",
    class_map: Dict[str, int] = None,
) -> Tuple[int, int]:
    """
    执行完整的 LabelMe → YOLO 转换流程。

    Args:
        labelme_dir: LabelMe JSON 根目录 (下含 train/ 和 val/)
        output_dir:  YOLO 标签输出根目录 (下含 train/ 和 val/)
        class_map:   类别名到ID的映射，为 None 时自动从 JSON 中提取

    Returns:
        (处理文件总数, 转换目标总数)
    """
    total_imgs, total_objs = 0, 0

    for split in ["train", "val"]:
        lm_split_dir = os.path.join(labelme_dir, split)
        out_split_dir = os.path.join(output_dir, split)

        if not os.path.isdir(lm_split_dir):
            print(f"  [SKIP] 目录不存在: {lm_split_dir}")
            continue

        # 自动构建类别映射 (取所有 split 的并集)
        if class_map is None:
            class_map = build_class_map(lm_split_dir)
            print(f"[INFO] 自动识别类别: {class_map}")

        print(f"[INFO] 正在转换 {split} 集...")
        n_imgs, n_objs = convert_directory(lm_split_dir, out_split_dir, class_map)
        total_imgs += n_imgs
        total_objs += n_objs
        print(f"  [OK] {split}: {n_imgs} 个文件, {n_objs} 个标注")

    print(f"\n[INFO] 转换完成: 共 {total_imgs} 个文件, {total_objs} 个标注")
    return total_imgs, total_objs


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LabelMe JSON → YOLO segmentation 标签转换")
    parser.add_argument("--labelme-dir", type=str, default="dataset/labelme",
                        help="LabelMe JSON 根目录 (默认: dataset/labelme)")
    parser.add_argument("--output-dir", type=str, default="dataset/labels",
                        help="YOLO 标签输出目录 (默认: dataset/labels)")
    parser.add_argument("--classes", type=str, default=None,
                        help="类别映射，格式: 'name:id,name:id' (默认: 自动识别)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # 解析类别映射
    class_map = None
    if args.classes:
        class_map = {}
        for pair in args.classes.split(","):
            cname, cid = pair.strip().split(":")
            class_map[cname] = int(cid)

    run(args.labelme_dir, args.output_dir, class_map)
