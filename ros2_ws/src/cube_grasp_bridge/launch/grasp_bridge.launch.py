"""
grasp_bridge.launch.py — 一键启动视觉+抓取控制
================================================

用法:
  # 最简（静态 TF 全为 0，即相机和机械臂坐标系重合）:
  ros2 launch cube_grasp_bridge grasp_bridge.launch.py

  # 指定相机到机械臂的静态变换:
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
        DeclareLaunchArgument('tf_tx', default_value='0.0',
                              description='静态 TF 平移 x (m)'),
        DeclareLaunchArgument('tf_ty', default_value='0.0',
                              description='静态 TF 平移 y (m)'),
        DeclareLaunchArgument('tf_tz', default_value='0.0',
                              description='静态 TF 平移 z (m)'),
        DeclareLaunchArgument('tf_qx', default_value='0.0',
                              description='静态 TF 四元数 x'),
        DeclareLaunchArgument('tf_qy', default_value='0.0',
                              description='静态 TF 四元数 y'),
        DeclareLaunchArgument('tf_qz', default_value='0.0',
                              description='静态 TF 四元数 z'),
        DeclareLaunchArgument('tf_qw', default_value='1.0',
                              description='静态 TF 四元数 w'),
        DeclareLaunchArgument('min_confidence', default_value='0.5',
                              description='YOLO 最低置信度'),

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
            }],
            condition=IfCondition(LaunchConfiguration('vision')),
        ),

        # ── 抓取控制节点 ──
        Node(
            package='cube_grasp_bridge',
            executable='grasp_controller',
            name='grasp_controller',
            output='screen',
            condition=IfCondition(LaunchConfiguration('controller')),
        ),

        LogInfo(msg='=== cube_grasp_bridge 已启动 ==='),
    ])
