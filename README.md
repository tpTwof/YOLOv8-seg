# 🤖 YOLO-grab

> 基于 YOLOv8 实例分割的机器人自主抓取系统

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-orange.svg)](https://docs.ros.org/en/humble/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8%2Fv11%2Fv26-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

通过 **Intel RealSense** 深度相机 + **YOLO 实例分割** 检测桌面目标物体，计算三维空间坐标，结合 **ROS2 + MoveIt2** 控制机械臂完成 "看到 → 定位 → 抓取" 全自主闭环。

---

## 🎬 演示

![Demo](docs/demo.gif)

> 完整抓取演示：YOLO 检测桌面方块 → 坐标变换 → MoveIt 规划 → 机械臂抓取

## 🧱 系统架构

```
┌──────────────────┐
│  RealSense D435   │  RGB 1920×1080 + Depth 1280×720
└────────┬─────────┘
         ▼
┌──────────────────┐
│ vision_frontend   │  YOLO-seg 推理 → 目标 mask 质心 (u,v)
│ (Python)          │  → 深度中位数 → 相机坐标系 3D 反投影
│                   │  → JSON 输出到 stdout
└────────┬─────────┘
         │ stdout JSON
         ▼
┌──────────────────┐
│ yolo_pose_pub     │  ROS2 节点: 读取 stdout JSON
│ (cube_grasp_      │  → TF2 坐标变换 (相机 → 机械臂基座)
│  bridge)          │  → 发布 PoseStamped 到 /grasp_target_pose
└────────┬─────────┘
         │ /grasp_target_pose
         ▼
┌──────────────────┐
│ grasp_controller  │  ROS2 节点: 订阅位姿
│ (cube_grasp_      │  → MoveIt IK 求解 → 关节轨迹
│  bridge)          │  → 5 步抓取状态机 → 机械臂执行
└──────────────────┘
```

---

## 📁 目录结构

```
YOLO-grab/
│
├── infer.py                        # 独立推理脚本 (图片/视频/摄像头/RealSense)
├── vision_frontend.py              # 视觉前端: RealSense + YOLO → 3D 目标 JSON
├── test_rs.py                      # RealSense 相机连通性测试
├── requirements.txt                # Python 依赖
│
├── docs/                           # 文档
│   ├── run.md                      # 完整抓取链路启动步骤 (中文)
│   └── gripper.md                  # 夹爪手动控制命令
│
├── scripts/                        # 数据集构建 & 训练工具链
│   ├── run_pipeline.py             # 一键数据集构建流水线
│   ├── extract_frames.py           # 视频抽帧 + 训练/验证集划分
│   ├── auto_label.py               # 使用已有模型自动标注
│   ├── yolo_to_labelme.py          # YOLO 标注 → LabelMe JSON (供人工校正)
│   ├── labelme_to_yolo.py          # LabelMe JSON → YOLO 标注 (校正后转回)
│   └── train.py                    # YOLOv8-seg 微调训练脚本
│
├── utils/                          # 工具库
│   ├── __init__.py
│   └── logger.py                   # 带时间戳的日志工具
│
├── dataset/                        # 数据集目录
│   ├── data.yaml                   # YOLO 数据集配置 (单类: cube)
│   ├── videos/                     # 原始录制视频
│   ├── images/  train/ val/        # 训练/验证图片
│   ├── labels/  train/ val/        # YOLO 分割标注
│   └── labelme/ train/ val/        # LabelMe JSON 标注 (人工校正)
│
└── ros2_ws/src/cube_grasp_bridge/  # ROS2 功能包
    ├── package.xml
    ├── setup.py
    ├── launch/grasp_bridge.launch.py   # 一键启动文件
    └── cube_grasp_bridge/
        ├── yolo_pose_publisher.py      # 视觉 → 位姿发布节点
        └── grasp_controller_node.py    # 5 步抓取控制节点
```

---

## ⚙️ 环境要求

### 软件依赖

| 类别 | 依赖 | 版本 |
|------|------|------|
| **Python** | Python | ≥ 3.10 |
| | ultralytics (YOLO) | ≥ 8.0 |
| | OpenCV | ≥ 4.5 |
| | NumPy | ≥ 1.20 |
| | pyrealsense2 | ≥ 2.50 |
| **ROS2** | ROS2 | Humble (or later) |
| | MoveIt2 | — |
| | rclpy, geometry_msgs, sensor_msgs | — |
| | tf2_ros, tf2_geometry_msgs | — |
| | moveit_msgs, control_msgs, trajectory_msgs | — |

### 硬件

| 硬件 | 型号/说明 |
|------|-----------|
| 深度相机 | Intel RealSense D435 (或同系列) |
| 机械臂 | 4-DOF 机械臂 + 夹爪 (关节: `joint1`–`joint4`) |
| 机械臂驱动 | OpenManipulator-X (或兼容 MoveIt2 的臂) |

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/YOLO-grab.git
cd YOLO-grab
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 准备模型权重

将训练好的 YOLO-seg 模型权重放在项目根目录：

| 文件 | 说明 |
|------|------|
| `best.pt` | YOLOv8-seg 分割模型 (放在项目根目录) |
| `yolo26n.pt` | 训练时使用的 YOLO 预训练权重 |

> 模型权重不包含在本仓库中，需自行训练或下载。

### 4. 测试硬件

```bash
# 测试 RealSense 相机
python test_rs.py

# 仅推理测试 (不涉及 ROS)
python infer.py --source realsense
```

### 5. 编译 ROS2 工作空间

```bash
cd ros2_ws
colcon build --packages-select cube_grasp_bridge
source install/setup.bash
```

### 6. 一键启动

```bash
# 启动完整系统 (视觉 + 抓取控制)
ros2 launch cube_grasp_bridge grasp_bridge.launch.py

# 仅视觉节点
ros2 launch cube_grasp_bridge grasp_bridge.launch.py controller:=false

# 仅抓取控制节点
ros2 launch cube_grasp_bridge grasp_bridge.launch.py vision:=false
```

> 📖 完整的链路上电启动步骤 (MoveIt → 硬件驱动 → 控制器 → YOLO) 请参阅 [docs/run.md](docs/run.md)

---

## 🎮 Launch 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vision` | bool | `true` | 是否启动视觉前端节点 |
| `controller` | bool | `true` | 是否启动抓取控制节点 |
| `world_frame` | str | `world` | 机械臂世界坐标系名称 |
| `model_path` | str | `""` | YOLO 模型路径 (空则使用默认) |
| `min_confidence` | float | `0.5` | YOLO 检测置信度阈值 |
| `flip_x` | bool | `true` | 发送 IK 前对目标 X 取反 |
| `smooth_window` | int | `5` | 坐标平滑窗口帧数 |
| `show_vision` | bool | `false` | 是否显示 OpenCV 可视化窗口 |
| `tf_tx`, `tf_ty`, `tf_tz` | float | 标定值 | 相机 → 机械臂基座的静态 TF 平移 (m) |
| `tf_qx`, `tf_qy`, `tf_qz`, `tf_qw` | float | 标定值 | 相机 → 机械臂基座的静态 TF 旋转 (四元数) |

---

## 🔬 抓取流程

控制器内部执行 **5 步状态机**：

```
1. 张开夹爪
   ↓
2. 移动到预抓取点 (目标正上方 8 cm)
   ↓
3. 下降到抓取点 (目标上方 2 cm)
   ↓
4. 闭合夹爪
   ↓
5. 抬升 (上移 ~12 cm)
```

视觉前端输出的检测结果为 JSON 格式：

```json
{
  "class": "cube",
  "confidence": 0.92,
  "x": 0.15,
  "y": -0.03,
  "z": 0.42,
  "yaw": 1.23,
  "u": 320,
  "v": 240
}
```

---

## 🏷️ 数据集 & 训练

### 一键构建数据集

从视频到训练就绪数据集：

```bash
python scripts/run_pipeline.py
```

该流水线自动执行：
1. 从 `dataset/videos/` 中抽取视频帧
2. 使用已有模型进行自动标注
3. 转换为 LabelMe JSON 格式 (供人工校正)
4. 生成 `data.yaml` 训练配置文件

### 人工校正标注

```bash
# 1. 自动标注后，转为 LabelMe 格式
python scripts/yolo_to_labelme.py

# 2. 在 LabelMe 中打开 dataset/labelme/ 下的 JSON 文件进行校正
#    (labelme: pip install labelme && labelme)

# 3. 校正完成后，转回 YOLO 格式
python scripts/labelme_to_yolo.py
```

### 训练模型

```bash
python scripts/train.py \
    --model best.pt \
    --data dataset/data.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

训练完成后，最佳权重保存在 `runs/segment/train/weights/best.pt`。

---

## 🔧 独立推理测试

不依赖 ROS，纯视觉调试：

```bash
# 图片推理
python infer.py --source path/to/image.jpg

# 视频推理
python infer.py --source path/to/video.mp4

# 内置摄像头
python infer.py --source camera

# RealSense (含深度图叠加)
python infer.py --source realsense --depth
```

所有推理结果保存在 `results/` 目录。

---

## 🧪 视觉前端独立测试

```bash
# 后台运行，JSON 输出到 stdout
python vision_frontend.py

# 带 OpenCV 可视化窗口
python vision_frontend.py --show

# 指定模型
python vision_frontend.py --model best.pt
```

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

---

## 📝 License

本项目采用 [MIT License](LICENSE)。

---

<p align="center">
  <b>Made with ❤️ for robotic grasping research</b>
</p>
