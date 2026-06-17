"""
grasp_controller_node — 抓取执行节点
=====================================
订阅 /grasp_target_pose (PoseStamped)，执行完整抓取流程：
  1. 打开夹爪
  2. 移动到预抓取点 (物体上方)
  3. 移动到抓取点
  4. 关闭夹爪
  5. 抬起

调用接口：
  /compute_ik                              (moveit_msgs/srv/GetPositionIK)
  /arm_controller/follow_joint_trajectory  (control_msgs/action/FollowJointTrajectory)
  /gripper_controller/gripper_cmd          (control_msgs/action/GripperCommand)
"""

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint


# ──────────────────────────────────────────────
# 配置常量 — 根据你的机械臂调整
# ──────────────────────────────────────────────

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4']

PRE_GRASP_OFFSET_Z = 0.08
GRASP_OFFSET_Z     = 0.02
LIFT_OFFSET_Z      = 0.12

GRIPPER_OPEN   = 0.03
GRIPPER_CLOSE  = 0.008
GRIPPER_EFFORT = 1.0

IK_LINK = 'end_effector_link'

DEFAULT_ORIENTATION = {'x': 0.0, 'y': 0.7071, 'z': 0.0, 'w': 0.7071}

# 超时 (秒)
IK_TIMEOUT      = 5.0
TRAJ_TIMEOUT    = 15.0
GRIPPER_TIMEOUT = 10.0

# 轮询间隔 (秒)
POLL_INTERVAL = 0.05


def wait_for_future(future, timeout: float, logger=None) -> bool:
    """
    轮询等待 rclpy future 完成。
    不调用 spin_until_future_complete，避免在多线程 executor 里卡死。
    """
    start = time.time()
    while not future.done():
        if time.time() - start > timeout:
            if logger:
                logger.warn(f'等待 future 超时 ({timeout}s)')
            return False
        time.sleep(POLL_INTERVAL)
    return True


class GraspControllerNode(Node):

    def __init__(self):
        super().__init__('grasp_controller')

        # ── 订阅目标位姿 ──
        self.create_subscription(
            PoseStamped, '/grasp_target_pose', self._on_grasp_target, 10)

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

        self._busy = False

        self.get_logger().info('=== 抓取控制器就绪，等待 /grasp_target_pose ===')

    # ──────────────────────────────────────────
    # 回调：收到目标位姿 → 开线程执行
    # ──────────────────────────────────────────
    def _on_grasp_target(self, msg: PoseStamped):
        if self._busy:
            self.get_logger().warn('正在抓取中，忽略新目标')
            return

        self._busy = True
        self.get_logger().info(
            f'收到目标: ({msg.pose.position.x:.3f}, '
            f'{msg.pose.position.y:.3f}, {msg.pose.position.z:.3f}) '
            f'frame={msg.header.frame_id}')

        # 在独立线程执行抓取，不阻塞订阅回调
        t = threading.Thread(target=self._run_grasp_thread, args=(msg,), daemon=True)
        t.start()

    def _run_grasp_thread(self, msg: PoseStamped):
        try:
            self._execute_grasp(msg)
        except Exception as e:
            self.get_logger().error(f'抓取失败: {e}')
        finally:
            self._busy = False

    # ──────────────────────────────────────────
    # 抓取主流程
    # ──────────────────────────────────────────
    def _execute_grasp(self, target: PoseStamped):
        px = target.pose.position.x
        py = target.pose.position.y
        pz = target.pose.position.z

        ox = target.pose.orientation.x
        oy = target.pose.orientation.y
        oz = target.pose.orientation.z
        ow = target.pose.orientation.w
        if abs(ox) + abs(oy) + abs(oz) + abs(ow) < 0.01:
            ox, oy, oz, ow = DEFAULT_ORIENTATION['x'], DEFAULT_ORIENTATION['y'], \
                              DEFAULT_ORIENTATION['z'], DEFAULT_ORIENTATION['w']

        # [1/5] 打开夹爪
        self.get_logger().info('[1/5] 打开夹爪')
        self._send_gripper(GRIPPER_OPEN)

        # [2/5] 预抓取点
        self.get_logger().info('[2/5] 移动到预抓取点')
        if not self._move_to(px, py, pz + PRE_GRASP_OFFSET_Z, ox, oy, oz, ow):
            self.get_logger().error('预抓取点失败')
            return

        # [3/5] 抓取点
        self.get_logger().info('[3/5] 下降到抓取点')
        if not self._move_to(px, py, pz + GRASP_OFFSET_Z, ox, oy, oz, ow):
            self.get_logger().error('抓取点失败')
            return

        # [4/5] 关闭夹爪
        self.get_logger().info('[4/5] 关闭夹爪')
        self._send_gripper(GRIPPER_CLOSE)
        time.sleep(0.5)

        # [5/5] 抬起
        self.get_logger().info('[5/5] 抬起')
        self._move_to(px, py, pz + LIFT_OFFSET_Z, ox, oy, oz, ow)

        self.get_logger().info('=== 抓取流程完成 ===')

    # ──────────────────────────────────────────
    # IK 求解 + 发送轨迹
    # ──────────────────────────────────────────
    def _move_to(self, x, y, z, ox, oy, oz, ow) -> bool:
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

        self.get_logger().info(
            f'IK 请求: pos=({x:.3f}, {y:.3f}, {z:.3f}) '
            f'orient=({ox:.3f}, {oy:.3f}, {oz:.3f}, {ow:.3f})')

        future = self._ik_client.call_async(req)
        if not wait_for_future(future, IK_TIMEOUT + 2.0, self.get_logger()):
            self.get_logger().error('IK 服务调用超时')
            return False

        result = future.result()
        if result is None:
            self.get_logger().error('IK 返回 None')
            return False
        if result.error_code.val != 1:
            self.get_logger().error(f'IK 求解失败, error_code={result.error_code.val}')
            return False

        joint_positions = []
        for name in JOINT_NAMES:
            try:
                idx = result.solution.joint_state.name.index(name)
                joint_positions.append(result.solution.joint_state.position[idx])
            except ValueError:
                self.get_logger().error(f'IK 结果中找不到关节 {name}')
                return False

        self.get_logger().info(f'IK 解: {[f"{p:.3f}" for p in joint_positions]}')
        return self._send_trajectory(joint_positions)

    # ──────────────────────────────────────────
    # 发送关节轨迹
    # ──────────────────────────────────────────
    def _send_trajectory(self, positions: list) -> bool:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * len(JOINT_NAMES)
        point.time_from_start.sec = 2
        point.time_from_start.nanosec = 0
        goal.trajectory.points = [point]

        self.get_logger().info(f'发送轨迹: {len(positions)} 个关节')

        send_future = self._arm_client.send_goal_async(goal)
        if not wait_for_future(send_future, TRAJ_TIMEOUT, self.get_logger()):
            self.get_logger().error('发送轨迹超时')
            return False

        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('轨迹被拒绝')
            return False

        result_future = goal_handle.get_result_async()
        if not wait_for_future(result_future, TRAJ_TIMEOUT, self.get_logger()):
            self.get_logger().error('轨迹执行超时')
            return False

        self.get_logger().info('轨迹执行完成')
        return True

    # ──────────────────────────────────────────
    # 发送夹爪命令
    # ──────────────────────────────────────────
    def _send_gripper(self, position: float) -> bool:
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = GRIPPER_EFFORT

        self.get_logger().info(f'发送夹爪: position={position}')

        send_future = self._gripper_client.send_goal_async(goal)
        if not wait_for_future(send_future, GRIPPER_TIMEOUT, self.get_logger()):
            self.get_logger().warn('夹爪 goal 发送超时')
            return False

        goal_handle = send_future.result()
        if goal_handle is None:
            self.get_logger().warn('夹爪 goal 返回 None')
            return False
        if not goal_handle.accepted:
            self.get_logger().warn('夹爪 goal 被拒绝')
            return False

        result_future = goal_handle.get_result_async()
        if not wait_for_future(result_future, GRIPPER_TIMEOUT, self.get_logger()):
            self.get_logger().warn('夹爪执行等待超时')
            return False

        result = result_future.result().result
        self.get_logger().info(
            f'夹爪结果: position={result.position:.4f}, '
            f'effort={result.effort:.2f}, '
            f'stalled={result.stalled}, '
            f'reached_goal={result.reached_goal}')
        return True


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
