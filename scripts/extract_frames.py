"""
视频抽帧脚本
=============
将目录中的视频文件按指定间隔抽取帧，保存为图片，并按比例划分 train/val。

用法:
    python scripts/extract_frames.py
    python scripts/extract_frames.py --video-dir dataset/videos --output-dir dataset/images --interval 5 --val-ratio 0.2

输出:
    dataset/images/train/   -- 训练图片
    dataset/images/val/     -- 验证图片
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2


# ──────────────────────────────────────────────
# 核心函数
# ──────────────────────────────────────────────

def extract_frames_from_video(video_path: str, output_dir: str, interval: int = 5) -> List[str]:
    """
    从单个视频中按间隔抽取帧并保存。

    Args:
        video_path: 视频文件路径
        output_dir: 图片输出目录
        interval:   每隔多少帧取一帧 (默认 5)

    Returns:
        保存的图片路径列表
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] 无法打开视频: {video_path}")
        return []

    video_name = Path(video_path).stem  # 例如 "01"
    saved_paths = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔抽帧
        if frame_idx % interval == 0:
            filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_paths.append(filepath)

        frame_idx += 1

    cap.release()
    print(f"  [OK] {video_name}: 共 {frame_idx} 帧, 抽取 {len(saved_paths)} 张")
    return saved_paths


def split_train_val(image_paths: List[str], train_dir: str, val_dir: str, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    将图片列表随机划分为训练集和验证集，并移动到对应目录。

    Args:
        image_paths: 图片路径列表
        train_dir:   训练集目录
        val_dir:     验证集目录
        val_ratio:   验证集比例 (默认 0.2)
        seed:        随机种子 (保证可复现)

    Returns:
        (训练集路径列表, 验证集路径列表)
    """
    random.seed(seed)
    shuffled = image_paths.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - val_ratio))
    train_paths = shuffled[:split_idx]
    val_paths = shuffled[split_idx:]

    # 移动文件到对应目录
    for path in train_paths:
        dest = os.path.join(train_dir, os.path.basename(path))
        os.rename(path, dest)

    for path in val_paths:
        dest = os.path.join(val_dir, os.path.basename(path))
        os.rename(path, dest)

    return train_paths, val_paths


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def run(video_dir: str, output_dir: str, interval: int = 5, val_ratio: float = 0.2) -> Tuple[int, int]:
    """
    执行完整的视频抽帧流程。

    Args:
        video_dir:   视频文件目录
        output_dir:  图片输出根目录 (下含 train/ 和 val/)
        interval:    抽帧间隔
        val_ratio:   验证集比例

    Returns:
        (训练集数量, 验证集数量)
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 收集所有视频文件
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    video_files = sorted([
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if Path(f).suffix.lower() in video_extensions
    ])

    if not video_files:
        print(f"[ERROR] 在 {video_dir} 中未找到视频文件")
        return 0, 0

    print(f"[INFO] 找到 {len(video_files)} 个视频，开始抽帧 (间隔={interval})...")

    # 先抽帧到临时目录，再划分
    tmp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    all_images = []
    for video_path in video_files:
        images = extract_frames_from_video(video_path, tmp_dir, interval)
        all_images.extend(images)

    print(f"\n[INFO] 共抽取 {len(all_images)} 张图片，按 {int((1-val_ratio)*100)}/{int(val_ratio*100)} 划分 train/val...")

    # 划分并移动
    train_paths, val_paths = split_train_val(all_images, train_dir, val_dir, val_ratio)

    # 清理临时目录
    os.rmdir(tmp_dir)

    print(f"[INFO] 训练集: {len(train_paths)} 张  |  验证集: {len(val_paths)} 张")
    return len(train_paths), len(val_paths)


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="视频抽帧 → train/val 图片数据集")
    parser.add_argument("--video-dir", type=str, default="dataset/videos",
                        help="视频文件目录 (默认: dataset/videos)")
    parser.add_argument("--output-dir", type=str, default="dataset/images",
                        help="图片输出目录 (默认: dataset/images)")
    parser.add_argument("--interval", type=int, default=5,
                        help="每隔多少帧取一帧 (默认: 5)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="验证集比例 (默认: 0.2)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args.video_dir, args.output_dir, args.interval, args.val_ratio)
