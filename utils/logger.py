"""
日志工具模块
将操作日志写入 what_do/ 目录，文件名采用时间戳命名。
"""

import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "what_do")


def write_log(content: str) -> str:
    """
    将日志内容写入 what_do/ 目录，文件名格式: YYYYMMDD_HHMMSS.log

    Args:
        content: 日志正文，建议用简洁的要点格式

    Returns:
        日志文件的完整路径
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    filepath = os.path.join(LOG_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

    return filepath
