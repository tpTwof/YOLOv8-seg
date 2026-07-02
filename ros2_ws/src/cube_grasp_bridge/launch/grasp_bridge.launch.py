"""
grasp_bridge.launch.py — 一键启动视觉+抓取控制
================================================

用法:
  # 使用默认手眼标定参数 (EYE_OUT_HAND):
  ros2 launch cube_grasp_bridge grasp_bridge.launch.py

  # 指定相机到机械臂的静态变换 (覆盖标定值):
  ros2 launch cube_grasp_bridge grasp_bridge.launch.py \
    tf_tx:=0.3 tf_ty:=0.0 tf_tz:=0.5

  # 只启动抓取控制器（手动发布 /grasp_target_pose 测试）:
  ros2 launch cube_grasp_bridge grasp_bridge.launch.py vision:=false

  # 只启动视觉（抓取控制器在别的终端跑）:
  ros2 launch cube_grasp_bridge grasp_bridge.launch.py controller:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # ── 参数 ──
        DeclareLaunchArgument('vision',    default_value='true',
                              description='是否启动视觉前端'),
        DeclareLaunchArgument('controller', default_value='true',
                              description='是否启动抓取控制器'),
        DeclareLaunchArgument('world_frame', default_value='world',
                              description='机械臂世界坐标系名'),
        DeclareLaunchArgument('model_path', default_value='',
                              description='YOLO 模型路径，空则用默认'),
        # 手眼标定默认值 (EYE_OUT_HAND): Camera → Base
        DeclareLaunchArgument('tf_tx', default_value='0.19659611',
                              description='静态 TF 平移 x (m)'),
        DeclareLaunchArgument('tf_ty', default_value='-0.03518756',
                              description='静态 TF 平移 y (m)'),
        DeclareLaunchArgument('tf_tz', default_value='-0.06704601',
                              description='静态 TF 平移 z (m)'),
        DeclareLaunchArgument('tf_qx', default_value='0.29339955',
                              description='静态 TF 四元数 x'),
        DeclareLaunchArgument('tf_qy', default_value='0.25657593',
                              description='静态 TF 四元数 y'),
        DeclareLaunchArgument('tf_qz', default_value='0.61945565',
                              description='静态 TF 四元数 z'),
        DeclareLaunchArgument('tf_qw', default_value='0.68143980',
                              description='静态 TF 四元数 w'),
        DeclareLaunchArgument('min_confidence', default_value='0.5',
                              description='YOLO 最低置信度'),
        DeclareLaunchArgument('show_vision', default_value='false',
                              description='是否显示 OpenCV 可视化窗口'),
        DeclareLaunchArgument('smooth_window', default_value='5',
                              description='平滑窗口大小 (帧数)，越大越稳定但延迟越高'),
        DeclareLaunchArgument('flip_x', default_value='true',
                              description='X 轴取反 (发送 IK 前对目标 X 取反)'),

        # ── 视觉节点 ──
        Node(
            package='cube_grasp_bridge',
            executable='yolo_pose_publisher',
            name='yolo_pose_publisher',
            output='screen',
            parameters=[{
                'world_frame': LaunchConfiguration('world_frame'),
                'model_path': LaunchConfiguration('model_path'),
                'static_tf': True,
                'tf_tx': LaunchConfiguration('tf_tx'),
                'tf_ty': LaunchConfiguration('tf_ty'),
                'tf_tz': LaunchConfiguration('tf_tz'),
                'tf_qx': LaunchConfiguration('tf_qx'),
                'tf_qy': LaunchConfiguration('tf_qy'),
                'tf_qz': LaunchConfiguration('tf_qz'),
                'tf_qw': LaunchConfiguration('tf_qw'),
                'min_confidence': LaunchConfiguration('min_confidence'),
                'show_vision': LaunchConfiguration('show_vision'),
                'smooth_window': LaunchConfiguration('smooth_window'),
            }],
            condition=IfCondition(LaunchConfiguration('vision')),
        ),

        # ── 抓取控制节点 ──
        Node(
            package='cube_grasp_bridge',
            executable='grasp_controller',
            name='grasp_controller',
            output='screen',
            parameters=[{
                'flip_x': LaunchConfiguration('flip_x'),
            }],
            condition=IfCondition(LaunchConfiguration('controller')),
        ),

        LogInfo(msg='=== cube_grasp_bridge 已启动 ==='),
    ])
