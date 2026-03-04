#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
import numpy as np
import casadi
import threading
import math


class MPCTracker(Node):
    def __init__(self):
        super().__init__('mpc_tracker')

        self.declare_parameter('desired_linear_vel', 0.22)
        self.declare_parameter('max_linear_vel',     0.4)
        self.declare_parameter('max_angular_vel',    0.5)
        self.declare_parameter('horizon',            6)
        self.declare_parameter('dt',                 0.1)
        self.declare_parameter('goal_tolerance',     0.2)
        self.declare_parameter('obstacle_threshold', 1.0)
        self.declare_parameter('control_hz',         20.0)

        self.v_des      = self.get_parameter('desired_linear_vel').value
        self.v_max      = self.get_parameter('max_linear_vel').value
        self.w_max      = self.get_parameter('max_angular_vel').value
        self.N          = int(self.get_parameter('horizon').value)
        self.dt_p       = self.get_parameter('dt').value
        self.goal_tol   = self.get_parameter('goal_tolerance').value
        self.obs_thresh = self.get_parameter('obstacle_threshold').value
        self.hz         = self.get_parameter('control_hz').value

        # exponential obstacle soft cost: w_obs * exp(-dist / obs_sigma)
        self.obs_sigma = 0.3
        self.w_obs     = 8.0
        self.n_obs_pts = 6

        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        self.full_path    = []
        self.path_lock    = threading.Lock()
        self.scan_lock    = threading.Lock()
        self.goal_reached = False

        self.front_min  = float('inf')
        self.left_min   = float('inf')
        self.right_min  = float('inf')
        self.obs_points = [(99.0, 99.0)] * self.n_obs_pts
        self.scan_count = 0

        self._build_casadi_solver()

        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self.cmd_pub       = self.create_publisher(Twist,        'cmd_vel',          10)
        self.lookahead_pub = self.create_publisher(PointStamped, 'mpc/target_point', 10)
        self.path_sub = self.create_subscription(Path,      'path', self.path_cb,  transient_qos)
        self.odom_sub = self.create_subscription(Odometry,  'odom', self.odom_cb,  10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_cb,  10)
        self.timer    = self.create_timer(1.0 / self.hz, self.control_loop)

        self.get_logger().info(f'MPCTracker started [obs_thresh={self.obs_thresh}m, solver=CasADi/IPOPT]')

    def _build_casadi_solver(self):
        N  = self.N
        dt = self.dt_p
        n  = self.n_obs_pts

        opti = casadi.Opti()

        U    = opti.variable(2, N)
        X    = opti.variable(3, N + 1)
        X0   = opti.parameter(3)
        Xref = opti.parameter(2, N)
        Obs  = opti.parameter(2, n)

        opti.subject_to(X[:, 0] == X0)
        for k in range(N):
            v  = U[0, k]
            w  = U[1, k]
            th = X[2, k]
            opti.subject_to(X[:, k + 1] == casadi.vertcat(
                X[0, k] + v * casadi.cos(th) * dt,
                X[1, k] + v * casadi.sin(th) * dt,
                X[2, k] + w * dt
            ))

        opti.subject_to(opti.bounded(0.0,         U[0, :], self.v_max))
        opti.subject_to(opti.bounded(-self.w_max, U[1, :], self.w_max))

        cost = 0.0

        for k in range(N):
            dx = X[0, k + 1] - Xref[0, k]
            dy = X[1, k + 1] - Xref[1, k]
            cost += 15.0 * (dx**2 + dy**2)

        for k in range(N):
            cost += 1.0 * (U[0, k] - self.v_des)**2
            cost += 0.1 * U[1, k]**2

        for k in range(N - 1):
            cost += 2.0 * (U[0, k + 1] - U[0, k])**2
            cost += 1.0 * (U[1, k + 1] - U[1, k])**2

        for k in range(N):
            for j in range(n):
                dist2 = (X[0, k + 1] - Obs[0, j])**2 + (X[1, k + 1] - Obs[1, j])**2
                dist  = casadi.sqrt(dist2 + 1e-6)
                cost += self.w_obs * casadi.exp(-dist / self.obs_sigma)

        opti.minimize(cost)

        opts = {
            'ipopt.print_level':    0,
            'ipopt.max_iter':       100,
            'ipopt.tol':            1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'print_time':           False,
        }
        opti.solver('ipopt', opts)

        self._opti   = opti
        self._U      = U
        self._X      = X
        self._X0     = X0
        self._Xref   = Xref
        self._Obs    = Obs
        self._u_prev = np.zeros((2, N))

        self.get_logger().info('CasADi/IPOPT solver ready')

    def path_cb(self, msg):
        with self.path_lock:
            self.full_path    = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
            self.goal_reached = False
        self.get_logger().info(f'Path received: {len(self.full_path)} points')

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z))

    def scan_cb(self, msg):
        ranges = msg.ranges
        n_r    = len(ranges)

        def sector_min(start, end):
            vals = [ranges[i % n_r] for i in range(start, end)
                    if math.isfinite(ranges[i % n_r])
                    and msg.range_min <= ranges[i % n_r] <= msg.range_max]
            return min(vals) if vals else float('inf')

        front = min(sector_min(0, 60), sector_min(300, 360))
        left  = sector_min(60, 120)
        right = sector_min(240, 300)

        robot_x   = self.x
        robot_y   = self.y
        robot_yaw = self.yaw

        pts = []
        for i, r in enumerate(ranges):
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            ox = robot_x + r * math.cos(robot_yaw + angle)
            oy = robot_y + r * math.sin(robot_yaw + angle)
            pts.append((r, ox, oy))

        pts.sort(key=lambda p: p[0])
        obs = [(ox, oy) for (_, ox, oy) in pts[:self.n_obs_pts]]
        while len(obs) < self.n_obs_pts:
            obs.append((robot_x + 99.0, robot_y + 99.0))

        with self.scan_lock:
            self.front_min  = front
            self.left_min   = left
            self.right_min  = right
            self.obs_points = obs

        self.scan_count += 1
        if self.scan_count % 10 == 0:
            d = self.obs_thresh

            def label(v):
                if v < d:       return 'BLOCKED'
                if v < d * 1.5: return 'warning'
                return                  'clear'

            self.get_logger().info(
                f'LIDAR  F={front:.2f}m({label(front)})'
                f'  L={left:.2f}m({label(left)})'
                f'  R={right:.2f}m({label(right)})'
                f'  pos=({self.x:.2f},{self.y:.2f})'
                f'  yaw={math.degrees(self.yaw):.1f}deg'
            )

    def nearest_path_idx(self, path):
        best, idx = float('inf'), 0
        for i, (px, py) in enumerate(path):
            d = math.hypot(self.x - px, self.y - py)
            if d < best:
                best, idx = d, i
        return idx

    def nearest_ahead_idx(self, path):
        best_dist = float('inf')
        idx = None
        cx = math.cos(self.yaw)
        cy = math.sin(self.yaw)
        for i, (px, py) in enumerate(path):
            dx = px - self.x
            dy = py - self.y
            if dx * cx + dy * cy > 0:
                d = math.hypot(dx, dy)
                if d < best_dist:
                    best_dist = d
                    idx = i
        if idx is None:
            idx = self.nearest_path_idx(path)
        return idx

    def run_mpc(self, path, obs_points):
        idx = self.nearest_ahead_idx(path)
        rem = path[idx:]
        if len(rem) < 2:
            self.publish_cmd(0.0, 0.0)
            return

        ref     = np.array([rem[min(i * 3, len(rem) - 1)] for i in range(self.N)]).T
        obs_arr = np.array(obs_points, dtype=float).reshape(self.n_obs_pts, 2).T

        self._opti.set_value(self._X0,   [self.x, self.y, self.yaw])
        self._opti.set_value(self._Xref, ref)
        self._opti.set_value(self._Obs,  obs_arr)

        u_init = np.hstack([self._u_prev[:, 1:], self._u_prev[:, -1:]])
        self._opti.set_initial(self._U, u_init)
        self._opti.set_initial(self._X[:, 0], [self.x, self.y, self.yaw])

        try:
            sol = self._opti.solve()
            u_opt        = sol.value(self._U)
            self._u_prev = u_opt
            vc = float(u_opt[0, 0])
            wc = float(u_opt[1, 0])
        except Exception:
            vc = self.v_des
            wc = 0.0

        self.publish_cmd(vc, wc)

        pt = PointStamped()
        pt.header.stamp    = self.get_clock().now().to_msg()
        pt.header.frame_id = 'odom'
        pt.point.x = float(ref[0, 0])
        pt.point.y = float(ref[1, 0])
        self.lookahead_pub.publish(pt)

    def control_loop(self):
        with self.path_lock:
            if not self.full_path:
                return
            path = list(self.full_path)

        if self.goal_reached:
            self.publish_cmd(0.0, 0.0)
            return

        goal = path[-1]
        if math.hypot(self.x - goal[0], self.y - goal[1]) < self.goal_tol:
            self.publish_cmd(0.0, 0.0)
            self.goal_reached = True
            self.get_logger().info('Goal reached')
            return

        with self.scan_lock:
            front      = self.front_min
            left       = self.left_min
            right      = self.right_min
            obs_points = list(self.obs_points)

        hard_threshold = 0.4
        hard_blocked   = (front < hard_threshold or
                          left  < hard_threshold or
                          right < hard_threshold)

        if hard_blocked:
            turn_left = left >= right
            w_cmd     = self.w_max if turn_left else -self.w_max
            closest   = min(front, left, right)
            v_cmd     = 0.0 if closest < 0.20 else 0.10
            direction = 'left' if turn_left else 'right'
            self.get_logger().info(
                f'[safety] obstacle at {closest:.2f}m, turning {direction}'
            )
            self.publish_cmd(v_cmd, w_cmd)
        else:
            self.run_mpc(path, obs_points)

    def publish_cmd(self, v, w):
        msg = Twist()
        msg.linear.x  = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(MPCTracker())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
