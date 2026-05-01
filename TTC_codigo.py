import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from nav_msgs.msg import Odometry
import math

class FusionAvoider(Node):

    def __init__(self):
        super().__init__('fusion_avoider')

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.img_debug_pub = self.create_publisher(Image, '/TTC_debug', 10)

        self.bridge = CvBridge()
        self.prev_gray = None
        self.p0 = None
        self.prev_scale = 0

        self.ttc = None
        self.turn_dir = 1
        self.front_dist = 10.0
        
        self.target_yaw = None
        self.current_yaw = 0.0
        self.prev_angle_error = 0.0

        self.fps = 30.0

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        if self.target_yaw is None:
            self.target_yaw = self.current_yaw

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isinf(ranges), 10.0, ranges)
        ranges = np.where(np.isnan(ranges), 10.0, ranges)

        n = len(ranges)
        left = np.min(ranges[int(0.66*n):])
        front = np.min(ranges[int(0.33*n):int(0.66*n)])
        right = np.min(ranges[:int(0.33*n)])

        self.front_dist = front

        if left > right:
            self.turn_dir = 1
        else:
            self.turn_dir = -1

    def get_scale(self, points):
        center = np.mean(points, axis=0)
        dist = np.linalg.norm(points - center, axis=1)
        return np.mean(dist)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.p0 is None:
            self.p0 = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)
            if self.p0 is not None:
                self.prev_scale = self.get_scale(self.p0.reshape(-1, 2))
                self.prev_gray = gray
            return

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.p0, None)

        if p1 is None:
            self.p0 = None
            return

        st = st.reshape(-1)
        good_new = p1[st == 1].reshape(-1, 2)

        for pt in good_new:
            x, y = pt.ravel()
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        if len(good_new) > 5:
            current_scale = self.get_scale(good_new)
            delta = current_scale - self.prev_scale

            if delta > 0.0001:
                self.ttc = current_scale / (delta * self.fps)
            else:
                self.ttc = None

            self.prev_scale = current_scale
            self.p0 = good_new.reshape(-1, 1, 2)
            self.prev_gray = gray
        else:
            self.p0 = None
            self.prev_gray = gray

        if self.ttc is not None:
            cv2.putText(frame, f"TTC: {self.ttc:.2f}s",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "TTC: N/A",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        cv2.putText(frame, f"Front: {self.front_dist:.2f}m",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 2)

        debug_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        self.img_debug_pub.publish(debug_msg)

        cv2.imshow("TTC_DEBUG", frame)
        cv2.waitKey(1)

        self.make_decision()

    def make_decision(self):
        cmd = Twist()
        slow_distance = 1.2
        turn_distance = 0.9
        front = self.front_dist

        angle_error = 0.0
        angular_correction = 0.0
        if self.target_yaw is not None:
            angle_error = self.target_yaw - self.current_yaw
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
            
            # GANHOS REDUZIDOS: Para abrir a curva e proteger a traseira
            kp = 0.4  
            kd = 0.1  
            angular_correction = (kp * angle_error) + (kd * (angle_error - self.prev_angle_error))
            
            # LIMITADOR: Impede que o robô gire mais que 0.5 rad/s no retorno
            angular_correction = max(min(angular_correction, 0.5), -0.5)
            self.prev_angle_error = angle_error

        if front < turn_distance:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6 * self.turn_dir 

        elif front < slow_distance:
            cmd.linear.x = 0.2 
            cmd.angular.z = 0.3 * self.turn_dir 

        elif self.ttc is not None and self.ttc < 3.0:
            if self.ttc < 1.0:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                cmd.linear.x = 0.1
                cmd.angular.z = angular_correction 

        else:
            if self.target_yaw is not None:
                # VELOCIDADE LINEAR ALTA: Ajuda a fazer curvas mais largas (menos fechadas)
                cmd.linear.x = 1.8 
                cmd.angular.z = angular_correction
            else:
                cmd.linear.x = 2.4
                cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = FusionAvoider()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
