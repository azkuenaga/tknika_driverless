import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseArray
from mirena_common.msg import EntityList, Entity
from tknika_interfaces.msg import ConeDetectionArray


class SensorFusionNode(Node):

    def __init__(self):
        super().__init__("sensor_fusion_node")

        self.latest_cam_detections = []
        self.lidar_cnt = 0
        self.cam_cnt = 0

        # Subscriptions
        self.sub_lidar = self.create_subscription(
            PoseArray, "/tknika/cones", self.lidar_callback, 10
        )
        self.sub_cam = self.create_subscription(
            ConeDetectionArray,
            "/tknika/camera/detections",
            self.cam_callback,
            qos_profile_sensor_data,
        )

        # Output publisher
        self.fused_pub = self.create_publisher(EntityList, "/fused/cones", 10)

        # Diagnostic timer
        self.create_timer(1.0, self.log_diagnostics)

        self.get_logger().info("Sensor Fusion Node initialized successfully.")

    def cam_callback(self, msg: ConeDetectionArray):
        self.cam_cnt += 1
        if hasattr(msg, "detections") and msg.detections:
            self.latest_cam_detections = msg.detections

    def lidar_callback(self, pose_msg: PoseArray):
        self.lidar_cnt += 1

        fused_msg = EntityList()
        fused_msg.header = pose_msg.header

        # Map YOLO class IDs to string names expected by string type field
        class_map = {0: "blue_cone", 1: "orange_cone", 2: "yellow_cone"}

        for pose in pose_msg.poses:
            fused_cone = Entity()

            # 1. geometry_msgs/Point position
            fused_cone.position.x = float(pose.position.x)
            fused_cone.position.y = float(pose.position.y)
            fused_cone.position.z = float(pose.position.z)

            # 2. string type (Default fallback: "blue_cone")
            fused_cone.type = "blue_cone"

            # 3. float64 confidence (Default fallback: 1.0 for LiDAR detection)
            fused_cone.confidence = 1.0

            # Match with camera detection if available
            if self.latest_cam_detections:
                for c_det in self.latest_cam_detections:
                    if hasattr(c_det, "class_id"):
                        c_id = int(c_det.class_id)
                        fused_cone.type = str(class_map.get(c_id, "blue_cone"))
                        if hasattr(c_det, "confidence"):
                            fused_cone.confidence = float(c_det.confidence)
                        break

            fused_msg.entities.append(fused_cone)

        self.fused_pub.publish(fused_msg)

    def log_diagnostics(self):
        self.get_logger().info(
            f"[DIAGNOSTICS] Rates (1s) -> LiDAR: {self.lidar_cnt} Hz | "
            f"Cam: {self.cam_cnt} Hz | Active Camera Detections: {len(self.latest_cam_detections)}"
        )
        self.lidar_cnt = 0
        self.cam_cnt = 0


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
