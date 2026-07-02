如果你要从 **开机后完全重启整个 YOLO→ROS2→机械臂抓取链路**，按照下面顺序来。

## 1. 关闭所有旧进程

先把之前残留的节点全部杀掉：

```bash
pkill -f grasp_controller
pkill -f yolo_pose_publisher
pkill -f move_group
pkill -f ros2_control_node
pkill -f robot_state_publisher
pkill -f rviz2
```

作用：清空旧的 ROS2 节点，避免 controller、MoveIt、TF 冲突。

---

## 2. 启动机械臂硬件

进入机械臂工作区：

```bash
source /opt/ros/humble/setup.bash
```

启动 OpenManipulator-X：

```bash
ros2 launch open_manipulator_x_bringup hardware.launch.py
```

确认控制器正常：

```bash
ros2 control list_controllers
```

应该看到：

```text
joint_state_broadcaster active
arm_controller active
gripper_controller active
```

---

## 3. 启动 MoveIt

新终端：

```bash
source /opt/ros/humble/setup.bash
```

启动 MoveIt：

```bash
ros2 launch open_manipulator_x_moveit_config moveit_core.launch.py
```

确认：

```bash
ros2 service list | grep compute_ik
```

应出现：

```text
/compute_ik
```

---

## 4. 启动抓取控制器

新终端：

```bash
cd ~/shit/YOLO-grab/ros2_ws
source install/setup.bash

ros2 run cube_grasp_bridge grasp_controller
```

应看到：

```text
等待 /compute_ik 服务...
/compute_ik 已连接
/arm_controller 已连接
=== 抓取控制器就绪 ===
```

---

## 5. 手动验证机械臂

先别启动 YOLO。

测试一个安全位置：

```bash
ros2 topic pub --times 1 --wait-matching-subscriptions 1 \
/grasp_target_pose geometry_msgs/msg/PoseStamped \
'{header: {frame_id: "world"}, pose: {position: {x: 0.15, y: 0.0, z: 0.15}, orientation: {x: 0.0, y: 0.707, z: 0.0, w: 0.707}}}'
```

如果看到：

```text
IK 解
发送轨迹
轨迹执行完成
```

并且机械臂运动，说明控制链路正常。

---

## 6. 启动 YOLO

新终端：

```bash
cd ~/shit/YOLO-grab
```

激活你原来的 YOLO 环境：

```bash
conda activate <你的环境名>
```

运行：

```bash
python vision_frontend.py
```

或者：

```bash
ros2 run cube_grasp_bridge yolo_pose_publisher
```

取决于你现在封装成什么形式。

---

## 7. 观察关键日志

YOLO端应该不断打印：

```text
发布目标:
world=(x,y,z)
```

控制器应该打印：

```text
收到目标
发送预抓取 IK
IK 解
发送轨迹
轨迹执行完成
```

---

### 当前最重要的检查

你之前已经验证过：

```text
手动发 Pose → 机械臂能动
```

所以如果重新启动后：

```text
YOLO发布目标
机械臂不动
```

那么问题一定在：

```text
YOLO坐标
↓
/grasp_target_pose
↓
grasp_controller
```

这一段。

重启后先执行：

```bash
ros2 topic echo /grasp_target_pose
```

然后把：

```text
1. yolo_pose_publisher 日志
2. grasp_controller 日志
3. ros2 topic echo /grasp_target_pose 输出
```

发出来，我们就能定位现在卡在哪一步。
