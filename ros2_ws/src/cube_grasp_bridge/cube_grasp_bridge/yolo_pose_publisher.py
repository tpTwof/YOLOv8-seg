"""
yolo_pose_publisher — 视觉前端 ROS2 封装
==========================================
把 vision_frontend.py 的输出转换成 ROS2 PoseStamped，发布到 /grasp_target_pose。

工作方式：
  - 运行 vision_frontend.py 作为子进程，读取其 stdout JSON
  - 从 JSON 中提取 position_camera (相机坐标系下的 3D 位置)
  - 通过 TF2 把相机坐标转换到 world/base_link 坐标系
  - 发布 geometry_msgs/PoseStamped 到 /grasp_target_pose

如果没有 TF（相机到机械臂的变换未配置），会用静态变换 fallback。
"""

import collections
import json
import math
import subprocess
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from tf2_ros import Buffer, TransformListener
import numpy as np

# vision_frontend.py 默认路径
import os
DEFAULT_VISION_FRONTEND = os.path.expanduser('~/shit/YOLO-grab/vision_frontend.py')

# 相机坐标系名（和 vision_frontend.py 里一致）
CAMERA_FRAME = 'camera_color_optical_frame'

# TF 等待超时
TF_TIMEOUT = 3.0

# 手眼标定矩阵 T_cam2base (EYE_OUT_HAND)
T_CAM2BASE = np.array([
    [0.10088699, -0.69368495, 0.71317811, 0.19659611],
    [0.99480199, 0.06038281, -0.08199345, -0.03518756],
    [0.01381392, 0.71774307, 0.69617100, -0.06704601],
    [0.0,        0.0,        0.0,         1.0       ],
])
# T_cam2base 对应的四元数 (x, y, z, w)，预计算避免运行时依赖 scipy
T_CAM2BASE_QUAT = (0.29339955, 0.25657593, 0.61945565, 0.68143980)


class YoloPosePublisher(Node):

    def __init__(self):
        super().__init__('yolo_pose_publisher')

        # ── 参数 ──
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('model_path', '')
        self.declare_parameter('static_tf', False)        # 默认关闭静态 TF，用标定矩阵直算
        # 手眼标定结果 (EYE_OUT_HAND): Camera → Base
        self.declare_parameter('tf_tx', 0.19659611)       # 静态 TF 平移 x
        self.declare_parameter('tf_ty', -0.03518756)      # 静态 TF 平移 y
        self.declare_parameter('tf_tz', -0.06704601)      # 静态 TF 平移 z
        self.declare_parameter('tf_qx', 0.29339955)       # 静态 TF 四元数 x
        self.declare_parameter('tf_qy', 0.25657593)       # 静态 TF 四元数 y
        self.declare_parameter('tf_qz', 0.61945565)       # 静态 TF 四元数 z
        self.declare_parameter('tf_qw', 0.68143980)       # 静态 TF 四元数 w
        self.declare_parameter('min_confidence', 0.5)     # 最低置信度
        self.declare_parameter('vision_frontend', DEFAULT_VISION_FRONTEND)  # vision_frontend.py 路径
        self.declare_parameter('show_vision', False)      # 是否显示 OpenCV 可视化窗口
        self.declare_parameter('smooth_window', 5)        # 平滑窗口大小 (帧数)

        self._world_frame = self.get_parameter('world_frame').value
        self._model_path = self.get_parameter('model_path').value
        self._use_static_tf = self.get_parameter('static_tf').value
        self._min_confidence = self.get_parameter('min_confidence').value
        self._smooth_window = self.get_parameter('smooth_window').value

        # ── 平滑滤波缓冲区 ──
        self._pos_buf = collections.deque(maxlen=self._smooth_window)
        self._yaw_buf = collections.deque(maxlen=self._smooth_window)

        # ── TF ──
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        if self._use_static_tf:
            self._publish_static_tf()

        # ── 发布器 ──
        self._pub = self.create_publisher(PoseStamped, '/grasp_target_pose', 10)

        # ── 启动 vision_frontend 子进程 ──
        self._proc = None
        self._thread = threading.Thread(target=self._run_vision, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f'=== YOLO Pose Publisher 已启动 ===\n'
            f'  world_frame: {self._world_frame}\n'
            f'  static_tf: {self._use_static_tf}\n'
            f'  发布到: /grasp_target_pose')

    def _publish_static_tf(self):
        """发布一个静态 TF: world → camera_color_optical_frame。"""
        tx = self.get_parameter('tf_tx').value
        ty = self.get_parameter('tf_ty').value
        tz = self.get_parameter('tf_tz').value
        qx = self.get_parameter('tf_qx').value
        qy = self.get_parameter('tf_qy').value
        qz = self.get_parameter('tf_qz').value
        qw = self.get_parameter('tf_qw').value

        self._static_broadcaster = StaticTransformBroadcaster(self)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._world_frame
        t.child_frame_id = CAMERA_FRAME
        t.transform.translation.x = tx
        t.transform.translation.y = ty
        t.transform.translation.z = tz
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self._static_broadcaster.sendTransform(t)
        self.get_logger().info(
            f'静态 TF: {self._world_frame} → {CAMERA_FRAME} '
            f'({tx:.3f}, {ty:.3f}, {tz:.3f})')

    def _run_vision(self):
        """在后台线程运行 vision_frontend.py，逐行读取 JSON。"""
        import shutil
        vision_path = self.get_parameter('vision_frontend').value
        # 优先用 PATH 里的 python（conda 激活后指向 conda 环境）
        python_bin = shutil.which('python') or sys.executable
        cmd = [python_bin, vision_path]
        if self.get_parameter('show_vision').value:
            cmd.append('--show')
        if self._model_path:
            cmd += ['--model', self._model_path]

        self.get_logger().info(f'启动视觉前端: {" ".join(cmd)}')

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # 行缓冲
            )
        except Exception as e:
            self.get_logger().error(f'启动 vision_frontend 失败: {e}')
            return

        # 在另一个线程读 stderr（避免阻塞）
        def read_stderr():
            for line in self._proc.stderr:
                line = line.strip()
                if line:
                    self.get_logger().info(f'[vision] {line}')

        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()

        # 主循环：读 stdout JSON
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            self._process_frame(data)

        self.get_logger().info('vision_frontend 进程已退出')

    def _process_frame(self, data: dict):
        """处理一帧视觉结果，转换并发布 PoseStamped。"""
        if not data.get('has_target'):
            # 丢失目标，清空缓冲区，停止发布
            self._pos_buf.clear()
            self._yaw_buf.clear()
            return

        target = data['target']
        if target is None:
            return

        # 置信度过滤
        conf = target.get('confidence', 0.0)
        if conf < self._min_confidence:
            return

        # 3D 位置
        pos_cam = target.get('position_camera')
        if pos_cam is None:
            self.get_logger().warn('目标无 3D 位置，跳过')
            return

        # 抓取角
        grasp_yaw = target.get('grasp_yaw', 0.0)

        # 构造相机坐标系下的 PoseStamped
        pose_cam = PoseStamped()
        pose_cam.header.stamp = self.get_clock().now().to_msg()
        pose_cam.header.frame_id = CAMERA_FRAME
        pose_cam.pose.position.x = pos_cam['x']
        pose_cam.pose.position.y = pos_cam['y']
        pose_cam.pose.position.z = pos_cam['z']

        # 用 grasp_yaw 构造 orientation（绕 Z 轴旋转）
        half_yaw = grasp_yaw / 2.0
        pose_cam.pose.orientation.x = 0.0
        pose_cam.pose.orientation.y = 0.0
        pose_cam.pose.orientation.z = math.sin(half_yaw)
        pose_cam.pose.orientation.w = math.cos(half_yaw)

        # 转换到 world 坐标系
        pose_world = self._transform_to_world(pose_cam)
        if pose_world is None:
            return

        # ── 中值平滑 ──
        p = pose_world.pose.position
        self._pos_buf.append([p.x, p.y, p.z])
        self._yaw_buf.append(grasp_yaw)

        if len(self._pos_buf) < self._smooth_window:
            # 缓冲区未满，暂不发布
            return

        # 取中位数
        pos_arr = np.array(self._pos_buf)
        median_pos = np.median(pos_arr, axis=0)
        median_yaw = float(np.median(self._yaw_buf))

        # 构造平滑后的 PoseStamped
        smooth = PoseStamped()
        smooth.header.stamp = self.get_clock().now().to_msg()
        smooth.header.frame_id = self._world_frame
        smooth.pose.position.x = float(median_pos[0])
        smooth.pose.position.y = float(median_pos[1])
        smooth.pose.position.z = float(median_pos[2])

        half_yaw = median_yaw / 2.0
        smooth.pose.orientation.x = 0.0
        smooth.pose.orientation.y = 0.0
        smooth.pose.orientation.z = math.sin(half_yaw)
        smooth.pose.orientation.w = math.cos(half_yaw)

        self._pub.publish(smooth)
        self.get_logger().info(
            f'发布目标: {target["class_name"]} ({conf:.2f}) '
            f'world=({median_pos[0]:.3f}, {median_pos[1]:.3f}, {median_pos[2]:.3f}) '
            f'[smoothed n={len(self._pos_buf)}]')

    def _transform_cam_to_base_direct(self, pose_cam: PoseStamped) -> PoseStamped:
        """用手眼标定矩阵直接将相机坐标变换到 base 坐标 (TF 不可用时的 fallback)。"""
        R = T_CAM2BASE[:3, :3]
        t = T_CAM2BASE[:3, 3]

        p_cam = np.array([pose_cam.pose.position.x,
                          pose_cam.pose.position.y,
                          pose_cam.pose.position.z])
        p_base = R @ p_cam + t

        # 变换姿态: q_base = q_cam2base * q_cam
        qx_tf, qy_tf, qz_tf, qw_tf = T_CAM2BASE_QUAT

        so = pose_cam.pose.orientation
        # q_out = q_tf * q_src
        ow = qw_tf*so.w - qx_tf*so.x - qy_tf*so.y - qz_tf*so.z
        ox = qw_tf*so.x + qx_tf*so.w + qy_tf*so.z - qz_tf*so.y
        oy = qw_tf*so.y - qx_tf*so.z + qy_tf*so.w + qz_tf*so.x
        oz = qw_tf*so.z + qx_tf*so.y - qy_tf*so.x + qz_tf*so.w

        pose_base = PoseStamped()
        pose_base.header.stamp = pose_cam.header.stamp
        pose_base.header.frame_id = self._world_frame
        pose_base.pose.position.x = float(p_base[0])
        pose_base.pose.position.y = float(p_base[1])
        pose_base.pose.position.z = float(p_base[2])
        pose_base.pose.orientation.x = float(ox)
        pose_base.pose.orientation.y = float(oy)
        pose_base.pose.orientation.z = float(oz)
        pose_base.pose.orientation.w = float(ow)

        return pose_base

    @staticmethod
    def _quat_to_rot(qx, qy, qz, qw):
        """四元数转 3x3 旋转矩阵。"""
        return np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
        ])

    def _transform_to_world(self, pose_cam: PoseStamped) -> PoseStamped:
        """把相机坐标系下的 PoseStamped 转换到 world 坐标系。"""
        try:
            transform = self._tf_buffer.lookup_transform(
                self._world_frame,           # 目标
                pose_cam.header.frame_id,     # 源
                rclpy.time.Time(),            # 最新
                timeout=Duration(seconds=TF_TIMEOUT),
            )
        except Exception as e:
            self.get_logger().warn(
                f'TF 变换失败 ({CAMERA_FRAME} → {self._world_frame}): {e}\n'
                f'使用手眼标定矩阵直接变换作为 fallback')
            return self._transform_cam_to_base_direct(pose_cam)

        t = transform.transform.translation
        r = transform.transform.rotation
        R = self._quat_to_rot(r.x, r.y, r.z, r.w)

        # 变换位置
        p_cam = np.array([pose_cam.pose.position.x,
                          pose_cam.pose.position.y,
                          pose_cam.pose.position.z])
        p_world = R @ p_cam + np.array([t.x, t.y, t.z])

        # 变换姿态（四元数乘法）
        # source orientation
        so = pose_cam.pose.orientation
        # transform orientation
        to = r
        # q_out = q_tf * q_src
        ow = to.w*so.w - to.x*so.x - to.y*so.y - to.z*so.z
        ox = to.w*so.x + to.x*so.w + to.y*so.z - to.z*so.y
        oy = to.w*so.y - to.x*so.z + to.y*so.w + to.z*so.x
        oz = to.w*so.z + to.x*so.y - to.y*so.x + to.z*so.w

        pose_world = PoseStamped()
        pose_world.header.stamp = pose_cam.header.stamp
        pose_world.header.frame_id = self._world_frame
        pose_world.pose.position.x = float(p_world[0])
        pose_world.pose.position.y = float(p_world[1])
        pose_world.pose.position.z = float(p_world[2])
        pose_world.pose.orientation.x = float(ox)
        pose_world.pose.orientation.y = float(oy)
        pose_world.pose.orientation.z = float(oz)
        pose_world.pose.orientation.w = float(ow)

        return pose_world

    def destroy_node(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
