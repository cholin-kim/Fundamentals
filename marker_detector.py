import cv2
import math
import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bridge = CvBridge()

class Marker_Detect():
    def __init__(self):

        rospy.init_node('image_listener')
        image_topic = "/camera/color/image_raw"

        self.distCoeffs = np.array([-0.0536506250500679, 0.06849482655525208, -0.0003799688129220158, 0.0006371866911649704, -0.021824028342962265])
        self.camMatrix = np.array([[640.7025756835938, 0.0, 655.5432739257812], [0.0, 639.7518310546875, 366.194580078125], [0.0, 0.0, 1.0]])

        rospy.Subscriber(image_topic, Image, self.image_callback)

        self.cv2_img = 0
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        rospy.wait_for_message(image_topic, Image)


    def image_callback(self, msg):
        self.cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")

    def find_aruco(self):
        arucoParam = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(self.arucoDict, arucoParam)
        corners, ids, rejected = detector.detectMarkers(self.cv2_img)
        return corners, ids, rejected


    def aruco_display(self, corners, ids):
        # frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame = self.cv2_img
        print("# of detected markers: ", len(corners))
        if len(corners) > 0:
            ids = ids.flatten()
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[0], 0.06, self.camMatrix, self.distCoeffs)

            transform_translation_x = round(tvec[0][0][0] * 100, 2)
            transform_translation_y = round(tvec[0][0][1] * 100, 2)
            transform_translation_z = round(tvec[0][0][2] * 100, 2)

            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvec[0]))[0]
            r = R.from_matrix(rotation_matrix[0:3, 0:3])
            quat = r.as_quat()

            transform_rotation_x = quat[0]
            transform_rotation_y = quat[1]
            transform_rotation_z = quat[2]
            transform_rotation_w = quat[3]

            roll_x, pitch_y, yaw_z = self.euler_from_quaternion(transform_rotation_x,transform_rotation_y,
                                                           transform_rotation_z, transform_rotation_w)

            roll_x = round(math.degrees(roll_x), 2)
            pitch_y = round(math.degrees(pitch_y), 2)
            yaw_z = round(math.degrees(yaw_z), 2)

            cv2.putText(frame, "translation_x: {}cm".format(transform_translation_x), (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            cv2.putText(frame, "translation_y: {}cm".format(transform_translation_y), (50, 85), cv2.FONT_ITALIC, 1,(255, 0, 0), 2)
            cv2.putText(frame, "translation_z: {}cm".format(transform_translation_z), (50, 120), cv2.FONT_ITALIC, 1,(255, 0, 0), 2)
            cv2.putText(frame, "roll_x: {}deg".format(roll_x), (50, 155), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            cv2.putText(frame, "pitch_y: {}deg".format(pitch_y), (50, 190), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            cv2.putText(frame, "yaw_z: {}deg".format(yaw_z), (50, 225), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, self.camMatrix, self.distCoeffs, rvec, tvec, 0.03, 1)

            return frame

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

if __name__ == '__main__':
    md = Marker_Detect()
    import copy
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        # cv2.namedWindow("image", flags=cv2.WINDOW_KEEPRATIO)
        corners, ids, _ = md.find_aruco()
        if len(corners) > 0:
            image = md.aruco_display(corners, ids)
            cv2.imshow("image", image)
            rate.sleep()
            cv2.waitKey(1)
        else:
            corners, ids, _ = md.find_aruco()
            continue
