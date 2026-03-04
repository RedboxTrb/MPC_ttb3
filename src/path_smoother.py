#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.interpolate import splprep, splev
import csv
import os


class PathSmoother(Node):
    def __init__(self):
        super().__init__('path_smoother')

        self.declare_parameter('path_file', 'src/nav/waypoints/waypoint.csv')
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('path_resolution', 0.05)
        self.declare_parameter('spline_smoothing', 0.0)

        self.path_file = self.get_parameter('path_file').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.resolution = self.get_parameter('path_resolution').get_parameter_value().double_value
        self.smoothing = self.get_parameter('spline_smoothing').get_parameter_value().double_value

        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.path_pub = self.create_publisher(Path, 'path', transient_qos)
        self.waypoints_pub = self.create_publisher(Path, 'waypoints', transient_qos)

        self._published = False
        self.create_timer(2.0, self.timed_publish)

    def timed_publish(self):
        if not self._published:
            self.load_and_publish()
            self._published = True

    def load_and_publish(self):
        waypoints = self.load_csv(self.path_file)
        if len(waypoints) < 2:
            self.get_logger().error('Need at least 2 waypoints')
            return

        self.publish_path(waypoints, self.waypoints_pub)

        smoothed = self.smooth_path(waypoints)
        self.publish_path(smoothed, self.path_pub)

        self.get_logger().info(
            f'Published {len(waypoints)} waypoints and {len(smoothed)} smoothed points'
        )

    def load_csv(self, filepath):
        waypoints = []
        if not os.path.exists(filepath):
            self.get_logger().error(f'File not found: {filepath}')
            return waypoints

        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    x = float(row[0].strip())
                    y = float(row[1].strip())
                    waypoints.append((x, y))
                except ValueError:
                    continue

        return waypoints

    def smooth_path(self, waypoints):
        if len(waypoints) < 3:
            return waypoints

        pts = np.array(waypoints)
        x = pts[:, 0]
        y = pts[:, 1]

        total_length = sum(
            np.hypot(x[i+1] - x[i], y[i+1] - y[i]) for i in range(len(x) - 1)
        )
        num_points = max(int(total_length / self.resolution), len(waypoints))

        try:
            tck, _ = splprep([x, y], s=self.smoothing, k=min(3, len(waypoints) - 1))
            u_fine = np.linspace(0, 1, num_points)
            x_smooth, y_smooth = splev(u_fine, tck)
            return list(zip(x_smooth.tolist(), y_smooth.tolist()))
        except Exception as e:
            self.get_logger().error(f'Spline fitting failed: {e}')
            return waypoints

    def publish_path(self, points, publisher):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for x, y in points:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathSmoother()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
