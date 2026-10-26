#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class LidarVisualizer(Node):

    def __init__(self):
        super().__init__("tknika_lidar_visualizer")

        self.subscription = self.create_subscription(
            PointCloud2,
            "/lidar/lidar",
            self.lidar_callback,
            qos_profile_sensor_data,
        )

        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(9, 9))

        self.scatter = self.ax.scatter(
            [], [],
            s=2,
            c=[],
            cmap="viridis",
            vmin=-1.0,
            vmax=1.0,
        )

        self.ax.set_xlim(-20, 20)
        self.ax.set_ylim(-20, 20)
        self.ax.set_aspect("equal")

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_title("Godot LiDAR - Top View")

        self.ax.grid(True)

        self.get_logger().info(
            "LiDAR visualizer started: /lidar/lidar"
        )

    def lidar_callback(self, msg):

        points = point_cloud2.read_points(
            msg,
            field_names=["x", "y", "z"],
            skip_nans=True,
        )

        # Convert the structured ROS point cloud into an Nx3 array.
        xyz = np.column_stack([
            points["x"],
            points["y"],
            points["z"],
        ])

        if len(xyz) == 0:
            return

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # Initial region of interest.
        # We will adjust these limits after examining the plot.
        mask = (
            (x > -20) & (x < 20) &
            (y > -20) & (y < 20) &
            (z > -1.0) & (z < 2.0)
        )

        xyz = xyz[mask]

        if len(xyz) == 0:
            return

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # Colour points by height.
        self.scatter.set_offsets(np.column_stack([x, y]))
        self.scatter.set_array(z)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def destroy_node(self):
        plt.close(self.fig)
        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = LidarVisualizer()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            plt.pause(0.001)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
