"""
数据集流水线脚本
=================
一键执行完整的数据集构建流程:
  1. 视频抽帧 → 图片
  2. 自动标注 → YOLO 标签
  3. YOLO → LabelMe JSON 转换
  4. 生成 data.yaml 训练配置

用法:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --interval 5 --val-ratio 0.2 --conf 0.25

完成后可用 LabelMe 打开 dataset/labelme/ 中的 JSON 文件进行人工微调。
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# 将项目根目录加入 sys.path，确保能导入 scripts 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_frames import run as extract_run
from scripts.auto_label import run as label_run
from scripts.yolo_to_labelme import run as convert_run


# ──────────────────────────────────────────────
# data.yaml 生成
# ──────────────────────────────────────────────

def generate_data_yaml(dataset_dir: str, class_names: Dict[int, str]) -> str:
    """
    生成 YOLO 训练所需的 data.yaml 配置文件。

    Args:
        dataset_dir:  数据集根目录
        class_names:  类别映射 {0: "cube"}

    Returns:
        生成的 yaml 文件路径
    """
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    lines = [
        "# YOLO 数据集配置文件 (自动生成)",
        f"path: {os.path.abspath(dataset_dir)}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]
    for cid, cname in sorted(class_names.items()):
        lines.append(f"  {cid}: {cname}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[INFO] 已生成 data.yaml: {yaml_path}")
    return yaml_path


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(
    video_dir: str = "dataset/videos",
    image_dir: str = "dataset/images",
    label_dir: str = "dataset/labels",
    labelme_dir: str = "dataset/labelme",
    dataset_dir: str = "dataset",
    model_path: str = "best.pt",
    interval: int = 5,
    val_ratio: float = 0.2,
    conf: float = 0.25,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """
    执行完整的数据集构建流水线。

    Args:
        video_dir:    视频目录
        image_dir:    图片输出目录
        label_dir:    YOLO 标签输出目录
        labelme_dir:  LabelMe JSON 输出目录
        dataset_dir:  数据集根目录
        model_path:   YOLO 模型路径
        interval:     抽帧间隔
        val_ratio:    验证集比例
        conf:         置信度阈值
        class_names:  类别映射
    """
    if class_names is None:
        class_names = {0: "cube"}

    print("=" * 60)
    print("  YOLO 数据集构建流水线")
    print("=" * 60)

    # ── 步骤 1: 视频抽帧 ──
    print("\n" + "─" * 40)
    print("步骤 1/4: 视频抽帧")
    print("─" * 40)
    extract_run(video_dir, image_dir, interval, val_ratio)

    # ── 步骤 2: 自动标注 ──
    print("\n" + "─" * 40)
    print("步骤 2/4: 自动标注 (YOLO segmentation)")
    print("─" * 40)
    label_run(image_dir, model_path, conf)

    # ── 步骤 3: YOLO → LabelMe ──
    print("\n" + "─" * 40)
    print("步骤 3/4: YOLO → LabelMe 转换")
    print("─" * 40)
    convert_run(image_dir, label_dir, labelme_dir, class_names)

    # ── 步骤 4: 生成 data.yaml ──
    print("\n" + "─" * 40)
    print("步骤 4/4: 生成训练配置")
    print("─" * 40)
    generate_data_yaml(dataset_dir, class_names)

    # ── 完成 ──
    print("\n" + "=" * 60)
    print("  流水线完成!")
    print("=" * 60)
    print(f"""
目录结构:
  {dataset_dir}/
  ├── images/
  │   ├── train/    ← 训练图片
  │   └── val/      ← 验证图片
  ├── labels/
  │   ├── train/    ← YOLO 标签
  │   └── val/      ← YOLO 标签
  ├── labelme/
  │   ├── train/    ← LabelMe JSON (可用于人工微调)
  │   └── val/      ← LabelMe JSON
  ├── data.yaml     ← YOLO 训练配置
  └── videos/       ← 原始视频 (不动)

下一步:
  1. 用 LabelMe 打开 dataset/labelme/ 中的 JSON，修正/补充标注
  2. 修改后的 JSON 可再转回 YOLO 格式用于训练
  3. 使用 data.yaml 进行 YOLO 训练
""")


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一键构建 YOLO 数据集 (抽帧 → 标注 → 转换)")
    parser.add_argument("--video-dir", type=str, default="dataset/videos",
                        help="视频目录 (默认: dataset/videos)")
    parser.add_argument("--model", type=str, default="best.pt",
                        help="YOLO 模型路径 (默认: best.pt)")
    parser.add_argument("--interval", type=int, default=5,
                        help="抽帧间隔 (默认: 5)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="验证集比例 (默认: 0.2)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
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

    run(
        video_dir=args.video_dir,
        model_path=args.model,
        interval=args.interval,
        val_ratio=args.val_ratio,
        conf=args.conf,
        class_names=class_names,
    )
