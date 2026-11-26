"""ROS 2 단일 통합 주행 노드 (/scan -> /drive).

- gt_free=True 모드에서는 /car_state/pose_GT, /car_state/odom_GT 토픽을 구독하지 않는다.
- publish()에서 속도를 항상 >= 0.0으로 클램프하여 '후진 불가!!' 규정을 이중으로 보장한다.
- gc.freeze()로 주행 중 파이썬 가비지 컬렉션 멈춤(70-85 ms)을 차단한다.
"""
import collections
import gc
import importlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def key(stamp):
    return (stamp.sec, stamp.nanosec)


class AvoidNode(Node):
    def __init__(self, spec, kwargs, rate):
        super().__init__('avoid_driver')
        mod, cls = spec.split(':')
        self.driver = getattr(importlib.import_module(mod), cls)(**kwargs)
        self.driver.reset()
        self.has_init_pose = callable(getattr(self.driver, 'set_initial_pose', None))
        self.gt_free = bool(getattr(self.driver, 'gt_free', False))
        self.scans = collections.deque(maxlen=10)
        self.poses = collections.OrderedDict()
        self.odoms = collections.OrderedDict()
        self.last_key = None
        self.failed = False
        self.ms, self.lat_ms, self.unmatched = [], [], 0

        self.create_subscription(LaserScan, '/scan', self.scan_cb, 20)
        if not self.gt_free:
            self.create_subscription(PoseStamped, '/car_state/pose_GT', self.pose_cb, 20)
            self.create_subscription(Odometry, '/car_state/odom_GT', self.odom_cb, 20)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.reset_cb, 10)
        self.pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.est_pub = (
            self.create_publisher(PoseStamped, '/lidar_loc/pose', 10)
            if getattr(self.driver, 'loc', None) is not None
            else None
        )
        self.create_timer(1.0 / rate, self.step)
        self.create_timer(5.0, self.report)
        self.last_call = None
        self.max_gap = 0.0
        gc.collect()
        gc.freeze()
        self.get_logger().info(f'avoid driver ready: {spec} {kwargs} at {rate} Hz')

    @staticmethod
    def _push(d, k, v):
        d[k] = v
        while len(d) > 30:
            d.popitem(last=False)

    def scan_cb(self, m):
        self.scans.append((key(m.header.stamp), np.asarray(m.ranges, dtype=np.float64)))

    def pose_cb(self, m):
        q = m.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self._push(self.poses, key(m.header.stamp), (m.pose.position.x, m.pose.position.y, yaw))

    def odom_cb(self, m):
        self._push(self.odoms, key(m.header.stamp), (m.twist.twist.linear.x, m.twist.twist.angular.z))

    def reset_cb(self, m):
        self.driver.reset()
        self.last_key = None
        if self.has_init_pose:
            p, q = m.pose.pose.position, m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.driver.set_initial_pose(p.x, p.y, yaw)
            self.scans.clear()
            self.get_logger().info(f'initialpose -> set_initial_pose({p.x:.3f}, {p.y:.3f}, {yaw:.3f})')
        if self.failed:
            self.failed = False
            self.get_logger().warn('initialpose -> clearing the failed state')
        self.get_logger().info('initialpose -> driver.reset()')

    def publish(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = max(0.0, float(speed))  # 후진 불가 규정 강제
        self.pub.publish(msg)

    def step(self):
        if self.failed:
            self.publish(0.0, 0.0)
            return
        if self.gt_free:
            match = self.scans[-1] if self.scans else None
        else:
            match = next(((k, r) for k, r in reversed(self.scans) if k in self.poses and k in self.odoms), None)
        if match is None:
            if self.scans:
                self.unmatched += 1
            return
        k, ranges = match
        if k == self.last_key:
            return
        self.last_key = k
        t = k[0] + k[1] * 1e-9
        t0 = time.perf_counter()
        if self.last_call is not None:
            self.max_gap = max(self.max_gap, t0 - self.last_call)
        self.last_call = t0
        try:
            if self.gt_free:
                steer, speed = self.driver.act(ranges, *(math.nan,) * 5, t)
            else:
                steer, speed = self.driver.act(ranges, *self.poses[k], *self.odoms[k], t)
        except Exception:
            self.failed = True
            self.get_logger().error('driver.act failed, stopping the car:\n' + traceback.format_exc())
            self.publish(0.0, 0.0)
            return
        self.publish(steer, speed)
        if self.est_pub is not None and getattr(self.driver.loc, 's', None) is not None:
            s = self.driver.loc.s
            m = PoseStamped()
            m.header.stamp.sec, m.header.stamp.nanosec = k
            m.header.frame_id = 'map'
            m.pose.position.x, m.pose.position.y = float(s[0]), float(s[1])
            m.pose.orientation.z = math.sin(float(s[2]) / 2)
            m.pose.orientation.w = math.cos(float(s[2]) / 2)
            self.est_pub.publish(m)
        self.ms.append((time.perf_counter() - t0) * 1e3)
        self.lat_ms.append((time.time() - t) * 1e3)

    def report(self):
        if self.ms:
            if self.gt_free:
                v = float(getattr(self.driver, 'v_est', math.nan))
            else:
                v = self.odoms[next(reversed(self.odoms))][0] if self.odoms else 0.0
            self.get_logger().info(
                f'v={v:.2f} act {np.mean(self.ms):.1f} ms (p99 {np.percentile(self.ms, 99):.1f}), '
                f'obs->cmd {np.mean(self.lat_ms):.1f} ms (p99 {np.percentile(self.lat_ms, 99):.1f}, '
                f'max {np.max(self.lat_ms):.0f}), max call gap {self.max_gap * 1e3:.0f} ms, '
                f'{len(self.ms)} calls, {self.unmatched} unmatched'
            )
            self.ms, self.lat_ms, self.unmatched, self.max_gap = [], [], 0, 0.0


def main():
    rclpy.init()
    spec = sys.argv[1]
    kwargs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    rate = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
    rclpy.spin(AvoidNode(spec, kwargs, rate))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
