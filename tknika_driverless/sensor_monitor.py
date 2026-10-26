#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge


class SensorMonitor(Node):

    def __init__(self):
        super().__init__("tknika_sensor_monitor")

        self.bridge = CvBridge()

        self.image_count = 0
        self.lidar_count = 0

        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.lidar_sub = self.create_subscription(
            PointCloud2,
            "/lidar/lidar",
            self.lidar_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info("Tknika sensor monitor started")
        self.get_logger().info("Camera: /camera/image_raw")
        self.get_logger().info("LiDAR: /lidar/lidar")

    def image_callback(self, msg):
        self.image_count += 1

        # Avoid logging every image.
        if self.image_count == 1 or self.image_count % 60 == 0:
            try:
                image = self.bridge.imgmsg_to_cv2(
                    msg,
                    desired_encoding="passthrough",
                )

                self.get_logger().info(
                    f"Camera: {msg.width}x{msg.height}, "
                    f"encoding={msg.encoding}, "
                    f"array_shape={image.shape}, "
                    f"frames_received={self.image_count}"
                )

            except Exception as exc:
                self.get_logger().error(
                    f"Camera conversion failed: {exc}"
                )

    def lidar_callback(self, msg):
        self.lidar_count += 1

        # Report the cloud structure on the first message.
        if self.lidar_count == 1:
            field_names = [field.name for field in msg.fields]

            self.get_logger().info(
                f"LiDAR frame: {msg.header.frame_id}"
            )
            self.get_logger().info(
                f"LiDAR dimensions: {msg.width}x{msg.height}"
            )
            self.get_logger().info(
                f"LiDAR fields: {field_names}"
            )

        # Log the number of valid points periodically.
        if self.lidar_count == 1 or self.lidar_count % 10 == 0:
            try:
                points = point_cloud2.read_points(
                    msg,
                    field_names=["x", "y", "z"],
                    skip_nans=True,
                )

                self.get_logger().info(
                    f"LiDAR: {len(points)} valid points, "
                    f"frames_received={self.lidar_count}"
                )

            except Exception as exc:
                self.get_logger().error(
                    f"LiDAR conversion failed: {exc}"
                )


def main(args=None):
    rclpy.init(args=args)

    node = SensorMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
