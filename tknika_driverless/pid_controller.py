import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
import numpy as np

# Core Mirena simulator messages
from mirena_common.msg import EntityList, CarControl


class PathFollowerPID(Node):

    def __init__(self):
        super().__init__("path_follower_pid")

        self.create_subscription(EntityList, "/fused/cones", self.control_loop, 10)
        self.control_pub = self.create_publisher(CarControl, "/control", 10)

        # PID Gains for steering
        self.Kp = 0.4
        self.Ki = 0.001
        self.Kd = 0.08

        self.prev_error = 0.0
        self.integral = 0.0

        # Maximum steering angle in radians (~25 degrees)
        self.max_steer_rad = 0.45  
        # Constant gas value between 0.0 and 1.0
        self.target_gas = 0.35  

        self.get_logger().info("PID Path Follower Controller initialized with gas/steer_angle interface.")

    def control_loop(self, msg: EntityList):
        blue_cones = []
        yellow_cones = []

        for cone in msg.entities:
            c_type = str(cone.type)
            if c_type in ["blue_cone", "0"]:
                blue_cones.append([cone.position.x, cone.position.y])
            elif c_type in ["yellow_cone", "2", "1"]:
                yellow_cones.append([cone.position.x, cone.position.y])

        # Calculate lateral cross-track error (y-offset)
        if blue_cones and yellow_cones:
            avg_blue = np.mean(blue_cones, axis=0)
            avg_yellow = np.mean(yellow_cones, axis=0)
            target = (avg_blue + avg_yellow) / 2.0
            error = target[1]
        elif blue_cones:
            avg_blue = np.mean(blue_cones, axis=0)
            error = avg_blue[1] - 1.5
        elif yellow_cones:
            avg_yellow = np.mean(yellow_cones, axis=0)
            error = avg_yellow[1] + 1.5
        else:
            error = 0.0

        # PID calculation
        self.integral += error
        derivative = error - self.prev_error
        steering_out = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error

        # Map PID output to steering angle in radians
        steer_rad = float(np.clip(steering_out, -self.max_steer_rad, self.max_steer_rad))

        # Populate CarControl message
        cmd = CarControl()
        cmd.header = msg.header  # Preserve timestamp
        cmd.gas = float(self.target_gas)  # Positive for gas [0.0 to 1.0]
        cmd.steer_angle = steer_rad       # Radians

        self.control_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerPID()
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
