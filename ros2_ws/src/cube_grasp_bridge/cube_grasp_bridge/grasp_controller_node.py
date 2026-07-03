"""
grasp_controller_node — 抓取执行节点
=====================================
订阅 /grasp_target_pose，状态机: 打开夹爪 → 预抓取 → 下降 → 关闭夹爪
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint


# ──────────────────────────────────────────────
# 配置常量
# ──────────────────────────────────────────────

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']

PRE_GRASP_OFFSET_Z = 0.08   # 预抓取点: 目标上方高度
GRASP_OFFSET_Z     = 0.02   # 抓取点: 略低于目标表面
TCP_Z_OFFSET       = -0.06  # TCP 修正: end_effector_link → 夹爪尖端的 z 偏差

GRIPPER_OPEN   = -0.02  # 负值 → 打开夹爪
GRIPPER_CLOSE  = 0.005   # 正值 → 关闭夹爪
GRIPPER_EFFORT = 1.0   # 夹爪最大力矩

IK_LINK = 'end_effector_link'

DEFAULT_ORIENTATION = {'x': 0.0, 'y': 0.7071, 'z': 0.0, 'w': 0.7071}

IK_TIMEOUT      = 5.0
TRAJ_TIMEOUT    = 15.0
GRIPPER_TIMEOUT = 10.0


class GraspControllerNode(Node):

    def __init__(self):
        super().__init__('grasp_controller')

        # ── 参数 ──
        self.declare_parameter('flip_x', True)  # X 轴取反 (发送 IK 前对目标 X 取反)
        self._flip_x = self.get_parameter('flip_x').value

        # ── 订阅目标位姿 ──
        self.create_subscription(
            PoseStamped, '/grasp_target_pose', self._on_grasp_target, 10)

        # ── 订阅关节状态 (给 IK 做种子，避免"转一圈") ──
        self._current_joints = {}
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 10)

        # ── IK 服务客户端 ──
        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.get_logger().info('等待 /compute_ik 服务...')
        self._ik_client.wait_for_service()
        self.get_logger().info('/compute_ik 已连接')

        # ── arm action 客户端 ──
        self._arm_client = ActionClient(
            self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.get_logger().info('等待 /arm_controller...')
        self._arm_client.wait_for_server()
        self.get_logger().info('/arm_controller 已连接')

        # ── gripper action 客户端 ──
        self._gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd')
        self.get_logger().info('等待 /gripper_controller...')
        self._gripper_client.wait_for_server()
        self.get_logger().info('/gripper_controller 已连接')

        # ── 状态机 ──
        self._busy = False
        self._step = 0
        self._pending_target = None
        self._pending_future = None     # 当前 async future
        self._step_start_time = 0.0     # 当前阶段开始时间（用于超时）
        self._joint_positions = []      # IK 解

        # 定时器 20Hz 非阻塞轮询
        self._timer = self.create_timer(0.05, self._step_timer_cb)

        self.get_logger().info(
            f'=== 抓取控制器就绪 (flip_x={self._flip_x})，等待 /grasp_target_pose ===')

    # ──────────────────────────────────────────
    # 回调：收到目标位姿
    # ──────────────────────────────────────────
    def _on_grasp_target(self, msg: PoseStamped):
        if self._busy:
            self.get_logger().warn('正在抓取中，忽略新目标')
            return

        # X 轴取反修正
        if self._flip_x:
            msg.pose.position.x = -msg.pose.position.x

        self.get_logger().info(
            f'收到目标: ({msg.pose.position.x:.3f}, '
            f'{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}) '
            f'frame={msg.header.frame_id}'
            f'{" [X翻转]" if self._flip_x else ""}')

        # 强制使用默认姿态（YOLO 姿态角可能 IK 无解）
        msg.pose.orientation.x = DEFAULT_ORIENTATION['x']
        msg.pose.orientation.y = DEFAULT_ORIENTATION['y']
        msg.pose.orientation.z = DEFAULT_ORIENTATION['z']
        msg.pose.orientation.w = DEFAULT_ORIENTATION['w']

        self._pending_target = msg
        self._busy = True
        self._step = 1

    def _on_joint_state(self, msg: JointState):
        """缓存当前关节位置，供 IK 做种子避免"转一圈"。"""
        for name, pos in zip(msg.name, msg.position):
            self._current_joints[name] = pos

    # ──────────────────────────────────────────
    # 定时器回调：非阻塞状态机 (4 个阶段)
    #   [1/6] 打开夹爪 → [2/6] 预抓取 → [3/6] 下降抓取 → [4/6] 关闭夹爪
    # 每步有独立的超时检测 (_step_start_time 在每阶段开始时刷新)
    # ──────────────────────────────────────────
    def _step_timer_cb(self):
        if self._step == 0:
            return

        if self._pending_target is None:
            self._reset('pending_target 丢失')
            return

        target = self._pending_target
        px = target.pose.position.x
        py = target.pose.position.y
        pz = target.pose.position.z
        ox = target.pose.orientation.x
        oy = target.pose.orientation.y
        oz = target.pose.orientation.z
        ow = target.pose.orientation.w

        pre_z  = pz + PRE_GRASP_OFFSET_Z + TCP_Z_OFFSET
        grab_z = pz + GRASP_OFFSET_Z     + TCP_Z_OFFSET

        # ── 阶段 1: 打开夹爪 ──
        if self._step == 1:
            self.get_logger().info('[1/6] 打开夹爪')
            self._pending_future = self._send_gripper_async(GRIPPER_OPEN)
            self._step = 2
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 2:
            if not self._pending_future.done():
                if self._check_timeout(GRIPPER_TIMEOUT, '夹爪打开超时'): self._reset(); return
                return
            gh = self._pending_future.result()
            if gh is None or not gh.accepted: self._reset('夹爪打开被拒'); return
            self._pending_future = gh.get_result_async()
            self._step = 3
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 3:
            if not self._pending_future.done():
                if self._check_timeout(GRIPPER_TIMEOUT, '夹爪打开超时'): self._reset(); return
                return
            self.get_logger().info('  夹爪已打开')
            self._step = 4
            # 继续执行步骤 4

        # ── 阶段 2: 移动到预抓取点 ──
        if self._step == 4:
            self.get_logger().info(f'[2/6] 预抓取 IK, z={pre_z:.3f}')
            self._pending_future = self._call_ik(px, py, pre_z, ox, oy, oz, ow)
            self._step = 5
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 5:
            if not self._pending_future.done():
                if self._check_timeout(IK_TIMEOUT, 'IK超时'): self._reset(); return
                return
            r = self._pending_future.result(); self._pending_future = None
            if r is None or r.error_code.val != 1: self._reset('IK失败'); return
            self._joint_positions = self._extract_joints(r)
            self._step = 6
            # 继续执行步骤 6

        if self._step == 6:
            self._pending_future = self._send_trajectory_async(self._joint_positions)
            self._step = 7
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 7:
            if not self._pending_future.done():
                if self._check_timeout(TRAJ_TIMEOUT, '轨迹超时'): self._reset(); return
                return
            gh = self._pending_future.result()
            if gh is None or not gh.accepted: self._reset('轨迹被拒'); return
            self._pending_future = gh.get_result_async()
            self._step = 8
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 8:
            if not self._pending_future.done():
                if self._check_timeout(TRAJ_TIMEOUT, '轨迹超时'): self._reset(); return
                return
            self.get_logger().info('  预抓取点到达')
            self._step = 9
            # 继续执行步骤 9

        # ── 阶段 3: 下降至抓取点 ──
        if self._step == 9:
            self.get_logger().info(f'[3/6] 抓取点 IK, z={grab_z:.3f}')
            self._pending_future = self._call_ik(px, py, grab_z, ox, oy, oz, ow)
            self._step = 10
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 10:
            if not self._pending_future.done():
                if self._check_timeout(IK_TIMEOUT, 'IK超时'): self._reset(); return
                return
            r = self._pending_future.result(); self._pending_future = None
            if r is None or r.error_code.val != 1: self._reset('IK失败'); return
            self._joint_positions = self._extract_joints(r)
            self._step = 11
            # 继续执行步骤 11

        if self._step == 11:
            self._pending_future = self._send_trajectory_async(self._joint_positions)
            self._step = 12
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 12:
            if not self._pending_future.done():
                if self._check_timeout(TRAJ_TIMEOUT, '轨迹超时'): self._reset(); return
                return
            gh = self._pending_future.result()
            if gh is None or not gh.accepted: self._reset('轨迹被拒'); return
            self._pending_future = gh.get_result_async()
            self._step = 13
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 13:
            if not self._pending_future.done():
                if self._check_timeout(TRAJ_TIMEOUT, '轨迹超时'): self._reset(); return
                return
            self.get_logger().info('  抓取点到达')
            self._step = 14
            # 继续执行步骤 14

        # ── 阶段 4: 关闭夹爪 ──
        if self._step == 14:
            self.get_logger().info('[4/6] 关闭夹爪')
            self._pending_future = self._send_gripper_async(GRIPPER_CLOSE)
            self._step = 15
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 15:
            if not self._pending_future.done():
                if self._check_timeout(GRIPPER_TIMEOUT, '夹爪关闭超时'): self._reset(); return
                return
            gh = self._pending_future.result()
            if gh is None or not gh.accepted: self._reset('夹爪关闭被拒'); return
            self._pending_future = gh.get_result_async()
            self._step = 16
            self._step_start_time = self.get_clock().now().nanoseconds * 1e-9
            return

        if self._step == 16:
            if not self._pending_future.done():
                if self._check_timeout(GRIPPER_TIMEOUT, '夹爪关闭超时'): self._reset(); return
                return
            self.get_logger().info('=== 抓取完成 ===')
            self._reset(None)

    # ──────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────
    def _call_ik(self, x, y, z, ox, oy, oz, ow):
        """异步发送 IK 请求，返回 future。用当前关节位置做种子，避免"转一圈"。"""
        req = GetPositionIK.Request()
        req.ik_request.group_name = 'arm'
        req.ik_request.ik_link_name = IK_LINK
        req.ik_request.pose_stamped.header.frame_id = 'world'
        req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.pose_stamped.pose.position.x = x
        req.ik_request.pose_stamped.pose.position.y = y
        req.ik_request.pose_stamped.pose.position.z = z
        req.ik_request.pose_stamped.pose.orientation.x = ox
        req.ik_request.pose_stamped.pose.orientation.y = oy
        req.ik_request.pose_stamped.pose.orientation.z = oz
        req.ik_request.pose_stamped.pose.orientation.w = ow
        req.ik_request.timeout.sec = int(IK_TIMEOUT)
        req.ik_request.timeout.nanosec = int((IK_TIMEOUT % 1) * 1e9)

        # 用当前关节位置做种子，求解器优先找"近路"
        seed_positions = []
        if self._current_joints:
            seed = JointState()
            for jn in JOINT_NAMES:
                if jn in self._current_joints:
                    seed.name.append(jn)
                    seed.position.append(self._current_joints[jn])
                    seed_positions.append(f'{jn}={self._current_joints[jn]:.3f}')
            if seed.name:
                req.ik_request.robot_state.joint_state = seed

        self.get_logger().info(
            f'  IK 请求: pos=({x:.3f}, {y:.3f}, {z:.3f}) '
            f'orient=({ox:.3f}, {oy:.3f}, {oz:.3f}, {ow:.3f})'
            f'{" seed=[" + ", ".join(seed_positions) + "]" if seed_positions else " (无种子)"}')
        return self._ik_client.call_async(req)

    def _extract_joints(self, ik_result):
        """从 IK 结果中提取关节值。"""
        joints = []
        for name in JOINT_NAMES:
            try:
                idx = ik_result.solution.joint_state.name.index(name)
                joints.append(ik_result.solution.joint_state.position[idx])
            except ValueError:
                self.get_logger().error(f'IK 结果中找不到关节 {name}')
                return None
        return joints

    def _send_trajectory_async(self, positions):
        """异步发送轨迹，返回 send_goal future。"""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(JOINT_NAMES)
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0
        goal.trajectory.points = [point]

        self.get_logger().info(f'  发送轨迹: {[f"{p:.3f}" for p in positions]}')
        return self._arm_client.send_goal_async(goal)

    def _send_gripper_async(self, position):
        """异步发送夹爪命令，返回 send_goal future。"""
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = GRIPPER_EFFORT
        self.get_logger().info(f'  发送夹爪: position={position}')
        return self._gripper_client.send_goal_async(goal)

    def _check_timeout(self, timeout_s: float, msg: str) -> bool:
        """检查是否超时，超时则打日志。"""
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._step_start_time > timeout_s:
            self.get_logger().error(f'{msg} ({timeout_s}s)')
            return True
        return False

    def _reset(self, reason: str = None):
        """重置状态机到空闲。"""
        if reason:
            self.get_logger().error(f'抓取中止: {reason}')
        self._step = 0
        self._pending_target = None
        self._pending_future = None
        self._joint_positions = []
        self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = GraspControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
