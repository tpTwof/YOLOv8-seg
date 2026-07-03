### 打开
```

ros2 action send_goal \
/gripper_controller/gripper_cmd \
control_msgs/action/GripperCommand \
"{command: {position: -0.01, max_effort: 1.0}}"

```

### 关闭
```
ros2 action send_goal \
/gripper_controller/gripper_cmd \
control_msgs/action/GripperCommand \
"{command: {position: 0.005, max_effort: 1.0}}"

```