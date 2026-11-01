#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data


from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseArray, Pose

from tf2_ros import Buffer, TransformListener, TransformException


class ConeDetector(Node):

    def __init__(self):
        super().__init__("tknika_cone_detector")

        # Detection parameters. We will tune these using real scans.
        self.declare_parameter("max_range", 12.0)
        self.declare_parameter("max_lateral", 6.0)
        self.declare_parameter("min_height", 0.05)
        self.declare_parameter("max_height", 0.80)
        self.declare_parameter("cluster_eps", 0.25)
        self.declare_parameter("min_samples", 4)

        self.max_range = float(
            self.get_parameter("max_range").value
        )
        self.max_lateral = float(
            self.get_parameter("max_lateral").value
        )
        self.min_height = float(
            self.get_parameter("min_height").value
        )
        self.max_height = float(
            self.get_parameter("max_height").value
        )
        self.cluster_eps = float(
            self.get_parameter("cluster_eps").value
        )
        self.min_samples = int(
            self.get_parameter("min_samples").value
        )

        # TF2 provides the transform from LiDAR to vehicle frame.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            "/lidar/lidar",
            self.cloud_callback,
            qos_profile_sensor_data,
        )

        self.cone_pub = self.create_publisher(
            PoseArray,
            "/tknika/cones",
            10,
        )
        self.filtered_cloud_pub = self.create_publisher(
            PointCloud2,
            "/tknika/debug/filtered_cloud",
            10,
        )

        self.clustered_cloud_pub = self.create_publisher(
            PointCloud2,
            "/tknika/debug/clustered_cloud",
            10,
        )

        self.processed_scans = 0

        self.get_logger().info("LiDAR cone detector started")
        self.get_logger().info("Input: /lidar/lidar")
        self.get_logger().info("Output: /tknika/cones")
        self.get_logger().info("Target frame: car/cog")

    @staticmethod
    def transform_points(points, transform):
        """
        Transform Nx3 points from the source frame into
        the target frame using a geometry_msgs Transform.
        """

        t = transform.translation
        q = transform.rotation

        # Quaternion components.
        x, y, z, w = q.x, q.y, q.z, q.w

        # Rotation matrix.
        rotation = np.array([
            [
                1 - 2 * (y*y + z*z),
                2 * (x*y - z*w),
                2 * (x*z + y*w),
            ],
            [
                2 * (x*y + z*w),
                1 - 2 * (x*x + z*z),
                2 * (y*z - x*w),
            ],
            [
                2 * (x*z - y*w),
                2 * (y*z + x*w),
                1 - 2 * (x*x + y*y),
            ],
        ], dtype=np.float32)

        translation = np.array(
            [t.x, t.y, t.z],
            dtype=np.float32,
        )

        # Row-vector representation: p_target = R*p_source + t.
        return points @ rotation.T + translation

    def publish_debug_cloud(self, points, labels, header):
        """
        Publish points in car/cog, colored by DBSCAN cluster.

        Label -1 represents noise and is colored gray.
        Each detected cluster receives a distinct color.
        """

        if len(points) == 0:
            return

        fields = [
            PointField(
                name="x",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="y",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="z",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1,
            ),
            PointField(
                name="rgb",
                offset=12,
                datatype=PointField.UINT32,
                count=1,
            ),
        ]

        def make_cloud(rgb_values):
            cloud_msg = PointCloud2()

            cloud_msg.header = header
            cloud_msg.header.frame_id = "car/cog"

            cloud_msg.height = 1
            cloud_msg.width = len(points)
            cloud_msg.fields = fields
            cloud_msg.is_bigendian = False
            cloud_msg.point_step = 16
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
            cloud_msg.is_dense = True

            data = np.empty(
                len(points),
                dtype=[
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("rgb", "<u4"),
                ],
            )

            data["x"] = points[:, 0]
            data["y"] = points[:, 1]
            data["z"] = points[:, 2]
            data["rgb"] = rgb_values

            cloud_msg.data = data.tobytes()

            return cloud_msg

        # White points show everything that passed filtering.
        white = np.full(
            len(points),
            0xFFFFFF,
            dtype=np.uint32,
        )

        self.filtered_cloud_pub.publish(make_cloud(white))

        # Gray indicates noise; clusters receive different colors.
        rgb_values = np.zeros(len(points), dtype=np.uint32)

        for i, label in enumerate(labels):
            if label == -1:
                rgb_values[i] = 0x808080
                continue

            # Generate well-separated colors using HSV.
            hue = (int(label) * 0.61803398875) % 1.0

            import colorsys

            r, g, b = colorsys.hsv_to_rgb(
                hue,
                1.0,
                1.0,
            )

            rgb_values[i] = (
                (int(r * 255) << 16)
                | (int(g * 255) << 8)
                | int(b * 255)
            )

        self.clustered_cloud_pub.publish(
            make_cloud(rgb_values)
        )

    def cloud_callback(self, msg):

        # Obtain the transform at the latest available time.
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                "car/cog",
                msg.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as exc:
            self.get_logger().warning(
                f"Waiting for LiDAR transform: {exc}"
            )
            return

        # Decode the organized PointCloud2 message.
        cloud = point_cloud2.read_points(
            msg,
            field_names=["x", "y", "z"],
            skip_nans=True,
        )

        points = np.column_stack((
            cloud["x"],
            cloud["y"],
            cloud["z"],
        )).astype(np.float32, copy=False)

        if len(points) == 0:
            return

        # Reduce computation by discarding distant points
        # before transforming them.
        source_range = np.linalg.norm(
            points[:, :2],
            axis=1,
        )

        points = points[source_range < self.max_range + 2.0]

        if len(points) == 0:
            return

        # Convert points into the vehicle body frame.
        points = self.transform_points(
            points,
            transform_stamped.transform,
        )

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # ROS convention:
        # X forward, Y left, Z up.
        #
        # Reject the immediate vicinity of the car,
        # points outside the useful region, and points
        # outside the approximate cone-height range.
        mask = (
            (x > 0.5)
            & (x < self.max_range)
            & (np.abs(y) < self.max_lateral)
            & (z > self.min_height)
            & (z < self.max_height)
        )

        candidates = points[mask]

        if len(candidates) == 0:
            self.publish_cones([], msg.header.stamp)
            return

        # Cluster horizontally: adjacent returns from the
        # same cone should be close in the XY plane.
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.min_samples,
            algorithm="kd_tree",
        )

        labels = clustering.fit_predict(
            candidates[:, :2]
        )

        # Publish diagnostic point clouds in car/cog.
        debug_header = msg.header
        debug_header.frame_id = "car/cog"

        self.publish_debug_cloud(
            candidates,
            labels,
            debug_header,
        )

        cones = []


        for label in set(labels):
            if label == -1:
                continue

            cluster = candidates[labels == label]

            if len(cluster) < self.min_samples:
                continue

            # Estimate the horizontal center robustly.
            center_x = float(np.median(cluster[:, 0]))
            center_y = float(np.median(cluster[:, 1]))

            # Approximate dimensions of the observed cluster.
            width_x = float(
                np.ptp(cluster[:, 0])
            )
            width_y = float(
                np.ptp(cluster[:, 1])
            )

            height = float(
                np.ptp(cluster[:, 2])
            )

            # Initial geometric rejection rules.
            # These are deliberately broad and need tuning.
            if max(width_x, width_y) > 0.8:
                continue

            if max(width_x, width_y) < 0.02:
                continue

            cones.append({
                "x": center_x,
                "y": center_y,
                "z": float(np.median(cluster[:, 2])),
                "points": len(cluster),
                "width": max(width_x, width_y),
                "height": height,
            })

        cones.sort(key=lambda cone: cone["x"])

        self.publish_cones(cones, msg.header.stamp)

        self.processed_scans += 1

        # if self.processed_scans % 10 == 0:
        #     self.get_logger().info(
        #         f"Scan {self.processed_scans}: "
        #         f"{len(candidates)} filtered points, "
        #         f"{len(cones)} candidate cones"
        #     )

        #     for index, cone in enumerate(cones):
        #         self.get_logger().info(
        #             f"  Cone {index}: "
        #             f"x={cone['x']:.2f} m, "
        #             f"y={cone['y']:.2f} m, "
        #             f"points={cone['points']}, "
        #             f"width={cone['width']:.2f} m"
        #         )

    def publish_cones(self, cones, stamp):

        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "car/cog"

        for cone in cones:
            pose = Pose()

            pose.position.x = cone["x"]
            pose.position.y = cone["y"]
            pose.position.z = cone["z"]

            # Identity orientation: these are point estimates.
            pose.orientation.w = 1.0

            msg.poses.append(pose)

        self.cone_pub.publish(msg)


def main(args=None):

    rclpy.init(args=args)
    node = ConeDetector()

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
