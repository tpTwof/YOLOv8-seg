# YOLO-grab

基于 YOLOv8 实例分割的机器人抓取系统。通过 Intel RealSense 深度相机检测目标物体（方块），计算其三维空间坐标，结合 ROS2 + MoveIt 控制机械臂完成自主抓取。

## 系统架构

```
┌─────────────────┐
│  RealSense 相机  │  RGB + Depth
└────────┬────────┘
         ▼
┌─────────────────┐
│ vision_frontend  │  YOLO-seg 推理 → 质心 (u,v) → 深度 → 三维反投影 → JSON
└────────┬────────┘
         │  stdout JSON
         ▼
┌─────────────────┐
│ yolo_pose_pub    │  ROS2 节点：读取 JSON → TF2 坐标变换 → PoseStamped
└────────┬────────┘
         │  /grasp_target_pose
         ▼
┌─────────────────┐
│ grasp_controller │  ROS2 节点：订阅位姿 → MoveIt IK → 关节轨迹 → 机械臂
└─────────────────┘
```

## 目录结构

```
YOLO-grab/
├── infer.py                    # 独立推理脚本（图片/视频/摄像头/RealSense）
├── vision_frontend.py          # 视觉前端：RealSense + YOLO → 3D 目标 JSON
├── test_rs.py                  # RealSense 相机连通性测试
├── requirements.txt            # Python 依赖
│
├── docs/                       # 文档
│   ├── run.md                  # 完整抓取链路启动步骤
│   └── gripper.md              # 夹爪相关说明
│
├── scripts/                    # 数据集构建与训练工具
│   ├── run_pipeline.py         # 一键数据集构建流水线
│   ├── extract_frames.py       # 视频抽帧 + 训练/验证集划分
│   ├── auto_label.py           # 使用已有模型自动标注
│   ├── yolo_to_labelme.py      # YOLO 标注 → LabelMe JSON（供人工校正）
│   ├── labelme_to_yolo.py      # LabelMe JSON → YOLO 标注（校正后转回）
│   └── train.py                # YOLOv8-seg 微调训练
│
├── utils/
│   ├── __init__.py
│   └── logger.py               # 带时间戳的日志工具
│
├── dataset/
│   ├── data.yaml               # YOLO 数据集配置（单类：cube）
│   ├── videos/                 # 原始录制视频
│   ├── images/train/           # 训练集图片
│   ├── images/val/             # 验证集图片
│   ├── labels/train/           # 训练集 YOLO 分割标注
│   ├── labels/val/             # 验证集 YOLO 分割标注
│   ├── labelme/train/          # LabelMe JSON 标注（用于人工校正）
│   └── labelme/val/
│
└── ros2_ws/src/cube_grasp_bridge/    # ROS2 功能包
    ├── package.xml
    ├── setup.py
    ├── launch/grasp_bridge.launch.py  # 启动文件
    └── cube_grasp_bridge/
        ├── yolo_pose_publisher.py     # 视觉 → 位姿发布节点
        └── grasp_controller_node.py   # 抓取控制节点
```

## 环境要求

**Python 依赖：**

```bash
pip install -r requirements.txt
```

- `ultralytics` — YOLOv8 训练与推理
- `opencv-python` — 图像/视频处理
- `numpy` — 数值计算
- `pyrealsense2` — Intel RealSense SDK

**ROS2 依赖（部署时需要）：**

- ROS2 Humble 或更高版本
- MoveIt2
- 依赖包：`rclpy`, `geometry_msgs`, `sensor_msgs`, `tf2_ros`, `tf2_geometry_msgs`, `moveit_msgs`, `control_msgs`, `trajectory_msgs`

**硬件：**

- Intel RealSense 深度相机（D435 等）
- 4 自由度机械臂 + 夹爪（关节名 `joint1`–`joint4`）

## 模型文件

模型权重不包含在本仓库中，需单独获取：

| 文件 | 说明 |
|------|------|
| `best.pt` | 训练好的 YOLOv8-seg 分割模型（放在项目根目录） |
| `yolo26n.pt` | YOLO26-nano 预训练权重（训练时使用） |

将 `.pt` 文件放到项目根目录后即可使用。

## 使用方式

### 1. 数据集构建

从视频到可训练数据集的一键流水线：

```bash
python scripts/run_pipeline.py
```

该命令依次执行：
1. 从 `dataset/videos/` 中抽帧
2. 使用已有模型自动标注
3. 转换为 LabelMe JSON 格式供人工校正
4. 生成 `data.yaml` 配置文件

**人工校正流程：**

```bash
# 1. 自动标注后，转为 LabelMe 格式
python scripts/yolo_to_labelme.py

# 2. 使用 LabelMe 工具打开 dataset/labelme/ 下的 JSON 文件进行校正

# 3. 校正完成后，转回 YOLO 格式
python scripts/labelme_to_yolo.py
```

### 2. 模型训练

```bash
python scripts/train.py \
    --model best.pt \
    --data dataset/data.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640
```

训练完成后，最佳权重保存在 `runs/segment/runs/segment/train/weights/best.pt`。

### 3. 独立推理测试

```bash
# 图片推理
python infer.py --source path/to/image.jpg

# 视频推理
python infer.py --source path/to/video.mp4

# 摄像头推理
python infer.py --source camera

# RealSense 推理（含深度可视化）
python infer.py --source realsense
```

结果保存在 `results/` 目录。

### 4. 视觉前端测试

```bash
# 仅运行视觉前端，输出 JSON 到 stdout
python vision_frontend.py

# 带可视化窗口
python vision_frontend.py --show
```

输出 JSON 格式：
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

### 5. ROS2 部署（完整系统）

```bash
# 编译 ROS2 工作空间
cd ros2_ws
colcon build --packages-select cube_grasp_bridge
source install/setup.bash

# 启动完整系统（视觉 + 抓取控制）
ros2 launch cube_grasp_bridge grasp_bridge.launch.py

# 仅启动视觉节点
ros2 launch cube_grasp_bridge grasp_bridge.launch.py controller:=false

# 仅启动抓取控制节点
ros2 launch cube_grasp_bridge grasp_bridge.launch.py vision:=false
```

**Launch 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vision` | `true` | 是否启动视觉节点 |
| `controller` | `true` | 是否启动抓取控制节点 |
| `model_path` | `` | YOLO 模型路径（空则用默认） |
| `min_confidence` | `0.5` | 检测置信度阈值 |
| `flip_x` | `true` | 发送 IK 前对目标 X 取反 |
| `smooth_window` | `5` | 平滑窗口帧数 |
| `show_vision` | `false` | 是否显示 OpenCV 可视化窗口 |
| `tf_tx`/`tf_ty`/`tf_tz` | 标定值 | 相机到机械臂的静态 TF 平移 |
| `tf_qx`/`tf_qy`/`tf_qz`/`tf_qw` | 标定值 | 相机到机械臂的静态 TF 旋转 |

## 抓取流程

控制器执行 5 步抓取序列：

1. **张开夹爪** — 准备抓取
2. **移动到预抓取点** — 目标正上方 8cm
3. **下降到抓取点** — 目标上方 2cm
4. **闭合夹爪** — 夹住物体
5. **抬升** — 上移 12cm 完成抓取

## License

MIT
