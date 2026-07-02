"""
YOLOv8-seg 微调训练脚本
========================
基于 best.pt 进行微调训练，支持命令行调节常见超参数。

用法:
    # 默认参数训练
    python scripts/train.py

    # 自定义参数
    python scripts/train.py --epochs 100 --batch 8 --imgsz 640 --lr 0.001

    # 从零训练 (不加载预训练权重)
    python scripts/train.py --model yolov8n-seg.pt --epochs 200

输出:
    runs/segment/train*/   -- 训练结果 (权重、日志、可视化)

注意:
    - 训练前请确保 dataset/data.yaml 中的路径正确
    - GPU 显存不足时减小 --batch 或 --imgsz
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


# ──────────────────────────────────────────────
# 默认超参数 (可被命令行覆盖)
# ──────────────────────────────────────────────

DEFAULTS = {
    "model": "best.pt",           # 预训练权重路径
    "data": "dataset/data.yaml",  # 数据集配置
    "epochs": 100,                 # 训练轮数
    "batch":8,                  # 批次大小 (显存不足时调小)
    "imgsz": 640,                 # 输入图片尺寸
    "lr0": 0.01,                  # 初始学习率
    "lrf": 0.01,                  # 最终学习率 = lr0 * lrf
    "optimizer": "auto",          # 优化器: SGD, Adam, AdamW, auto
    "patience": 20,               # 早停轮数 (无提升则停止)
    "device": "",                 # 设备: 0, 0,1, cpu (空=自动选)
    "workers": 8,                 # 数据加载线程数
    "project": "runs/segment",    # 输出目录
    "name": "train",              # 实验名称
    "exist_ok": False,            # 是否覆盖已有实验
    "resume": False,              # 是否从上次中断继续
    "seed": 42,                   # 随机种子
    "cache": False,               # 缓存图片到内存 (加速但占内存)
}


# ──────────────────────────────────────────────
# 核心训练函数
# ──────────────────────────────────────────────

def train(args: argparse.Namespace) -> str:
    """
    执行 YOLOv8-seg 微调训练。

    Args:
        args: 命令行参数

    Returns:
        训练输出目录路径
    """
    print("=" * 50)
    print("  YOLOv8-seg 微调训练")
    print("=" * 50)
    print(f"  模型:     {args.model}")
    print(f"  数据集:   {args.data}")
    print(f"  轮数:     {args.epochs}")
    print(f"  批次:     {args.batch}")
    print(f"  图片尺寸: {args.imgsz}")
    print(f"  学习率:   {args.lr0}")
    print(f"  优化器:   {args.optimizer}")
    print(f"  设备:     {args.device or '自动'}")
    print("=" * 50)


    # 加载模型
    model = YOLO(args.model)

    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        optimizer=args.optimizer,
        patience=args.patience,
        device=args.device or None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        seed=args.seed,
        cache=args.cache,
        # 数据增强参数 (可根据需要取消注释调整)
        # hsv_h=0.015,       # 色调增强
        # hsv_s=0.7,         # 饱和度增强
        # hsv_v=0.4,         # 亮度增强
        # degrees=10.0,      # 旋转角度
        # translate=0.1,     # 平移
        # scale=0.5,         # 缩放
        # fliplr=0.5,        # 水平翻转概率
        # mosaic=1.0,        # Mosaic 增强概率
    )

    # 输出结果路径
    save_dir = str(results.save_dir) if hasattr(results, "save_dir") else args.project + "/" + args.name
    print(f"\n[INFO] 训练完成，结果保存在: {save_dir}")
    print(f"[INFO] 最佳权重: {save_dir}/weights/best.pt")
    print(f"[INFO] 最终权重: {save_dir}/weights/last.pt")
    return save_dir


# ──────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv8-seg 微调训练")

    # 路径
    parser.add_argument("--model", type=str, default=DEFAULTS["model"],
                        help=f"预训练权重 (默认: {DEFAULTS['model']})")
    parser.add_argument("--data", type=str, default=DEFAULTS["data"],
                        help=f"数据集配置 (默认: {DEFAULTS['data']})")
    parser.add_argument("--project", type=str, default=DEFAULTS["project"],
                        help=f"输出目录 (默认: {DEFAULTS['project']})")
    parser.add_argument("--name", type=str, default=DEFAULTS["name"],
                        help=f"实验名称 (默认: {DEFAULTS['name']})")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"],
                        help=f"训练轮数 (默认: {DEFAULTS['epochs']})")
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"],
                        help=f"批次大小 (默认: {DEFAULTS['batch']})")
    parser.add_argument("--imgsz", type=int, default=DEFAULTS["imgsz"],
                        help=f"输入图片尺寸 (默认: {DEFAULTS['imgsz']})")
    parser.add_argument("--lr0", type=float, default=DEFAULTS["lr0"],
                        help=f"初始学习率 (默认: {DEFAULTS['lr0']})")
    parser.add_argument("--lrf", type=float, default=DEFAULTS["lrf"],
                        help=f"最终学习率系数 (默认: {DEFAULTS['lrf']})")
    parser.add_argument("--optimizer", type=str, default=DEFAULTS["optimizer"],
                        help=f"优化器 (默认: {DEFAULTS['optimizer']})")
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"],
                        help=f"早停轮数 (默认: {DEFAULTS['patience']})")

    # 环境
    parser.add_argument("--device", type=str, default=DEFAULTS["device"],
                        help="设备: 0, 0,1, cpu (默认: 自动)")
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"],
                        help=f"数据加载线程数 (默认: {DEFAULTS['workers']})")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
                        help=f"随机种子 (默认: {DEFAULTS['seed']})")
    parser.add_argument("--cache", action="store_true",
                        help="缓存图片到内存 (加速但占内存)")

    # 控制
    parser.add_argument("--exist-ok", action="store_true",
                        help="覆盖已有同名实验")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断继续训练")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
