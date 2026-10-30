#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from ultralytics import YOLO
from tknika_interfaces.msg import (
    ConeDetection,
    ConeDetectionArray,
)


class CameraDetector(Node):

    def __init__(self):
        super().__init__("tknika_camera_detector")

        self.declare_parameter(
            "model_path",
            "~/mirena_ws/src/tknika_driverless/models/cone_detection_model.pt",
        )
        self.declare_parameter("confidence", 0.40)
        self.declare_parameter("image_size", 640)

        model_path = str(
            Path(
                self.get_parameter("model_path").value
            ).expanduser()
        )

        confidence = float(
            self.get_parameter("confidence").value
        )

        self.image_size = int(
            self.get_parameter("image_size").value
        )

        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"YOLO model not found: {model_path}"
            )

        self.get_logger().info(
            f"Loading YOLO model: {model_path}"
        )

        self.model = YOLO(model_path)
        self.confidence = confidence

        self.get_logger().info(
            f"Model classes: {self.model.names}"
        )

        self.image_pub = self.create_publisher(
            Image,
            "/tknika/camera/detections_image",
            qos_profile_sensor_data,
        )
        self.detections_pub = self.create_publisher(
            ConeDetectionArray,
            "/tknika/camera/detections",
            qos_profile_sensor_data,
        )


        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.processed_frames = 0

        self.get_logger().info(
            "YOLO camera detector started"
        )
        self.get_logger().info(
            "Input: /camera/image_raw"
        )
        self.get_logger().info(
            "Output: /tknika/camera/detections_image"
        )
        self.get_logger().info(
            "Output detections: /tknika/camera/detections"
        )

    @staticmethod
    def image_to_bgr(msg):
        """Convert a ROS image into an OpenCV-compatible BGR array."""

        if msg.encoding not in ("rgba8", "rgb8", "bgr8"):
            raise ValueError(
                f"Unsupported image encoding: {msg.encoding}"
            )

        channels = 4 if msg.encoding == "rgba8" else 3

        image = np.frombuffer(
            msg.data,
            dtype=np.uint8,
        ).reshape(
            msg.height,
            msg.width,
            channels,
        )

        if msg.encoding == "bgr8":
            return image.copy()

        # RGB/RGBA -> BGR.
        return image[:, :, :3][:, :, ::-1].copy()

    def image_callback(self, msg):
        try:
            frame = self.image_to_bgr(msg)

            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.image_size,
                verbose=False,
            )

            result = results[0]

            # ------------------------------------------------
            # 1. CREATE STRUCTURED DETECTION MESSAGE
            # ------------------------------------------------

            detections_msg = ConeDetectionArray()
            
            # Preserve the original camera timestamp and frame.
            detections_msg.header = msg.header

            # Extract every detected cone.
            for box in result.boxes:
                
                detection = ConeDetection()
                
                # Cone classification.
                class_id = int(box.cls.item())
                
                detection.class_id = class_id
                
                detection.class_name = str(
                    result.names[class_id]
                )
                
                # Detection confidence.
                detection.confidence = float(
                    box.conf.item()
                )

                # Bounding box coordinates.
                x_min, y_min, x_max, y_max = (
                    box.xyxy[0].tolist()
                )

                detection.x_min = float(x_min)
                detection.y_min = float(y_min)
                detection.x_max = float(x_max)
                detection.y_max = float(y_max)

                # Add detection to the array.
                detections_msg.detections.append(
                    detection
                )

                # Publish structured detections.
                self.detections_pub.publish(
                    detections_msg
                )
            # ------------------------------------------------
            # 2. CREATE ANNOTATED IMAGE - Rvizen araztatzeko
            # ------------------------------------------------

            # Ultralytics returns an annotated BGR image.
            annotated = result.plot()

            output = Image()
            output.header = msg.header
            output.height = annotated.shape[0]
            output.width = annotated.shape[1]
            output.encoding = "bgr8"
            output.is_bigendian = 0
            output.step = output.width * 3
            output.data = annotated.tobytes()

            self.image_pub.publish(output)

            # ------------------------------------------------
            # 3. LOGGING
            # ------------------------------------------------

            self.processed_frames += 1

            if self.processed_frames % 30 == 0:
                boxes = result.boxes

                self.get_logger().info(
                    f"Frames processed: {self.processed_frames}, "
                    f"detections: {len(boxes)}"
                )

                for box in boxes:
                    class_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    class_name = result.names[class_id]

                    self.get_logger().info(
                        f"  {class_name}: "
                        f"confidence={confidence:.2f}"
                    )

        except Exception as exc:
            self.get_logger().error(
                f"Camera inference failed: {exc}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = None

    try:
        node = CameraDetector()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exc:
        print(f"Camera detector failed: {exc}")
        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
