import copy
import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R

class Aruco_Detect():
    def __init__(self):

        image_topic = "/camera/color/image_raw"
        aruco_pose_topic = "/aruco/pose"
        aruco_frame_img_topic = "/aruco/frame_img"

        self.distCoeffs = np.array(
            [-0.0536506250500679, 0.06849482655525208, -0.0003799688129220158, 0.0006371866911649704, -0.021824028342962265])
        self.camMatrix = np.array(
            [384.1087646484375, 0.0, 329.32598876953125, 0.0, 383.5387878417969, 243.7167510986328, 0.0, 0.0, 1.0]).reshape((3, 3))

        ## Publisher ##
        self.pub_aruco_pose = rospy.Publisher(aruco_pose_topic, PoseArray, queue_size=1)
        self.pub_aruco_frame_img = rospy.Publisher(aruco_frame_img_topic, Image, queue_size=1)

        ## Utils ##
        self.bridge = CvBridge()

        ## Subscriber ##
        rospy.Subscriber(image_topic, Image, self.image_callback)

        ## Initialize ##
        self.cv2_img = 0
        self.corners = 0
        self.ids = 0
        self.rejected = 0
        self.aruco_visualization = 0
        self.detect_flag = False

        rospy.wait_for_message(image_topic, Image)

        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        arucoParam = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(arucoDict, arucoParam)


    def image_callback(self, msg):
        self.cv2_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frame = copy.deepcopy(self.cv2_img)

    def find_aruco(self):
        self.corners, self.ids, self.rejected = self.detector.detectMarkers(self.cv2_img)
        print("Detected markers: ", self.ids, "#", len(self.corners))

        if len(self.corners) > 0 and np.max(self.ids) <= 5 and np.min(self.ids) >= 0:
            self.ids = self.ids.flatten()  ## {NoneType} object has no attribute 'flatten', convert to shape (n,)
            self.detect_flag = True
            self.publish_aruco()

        else:
            self.detect_flag = False
            print("searching again")
            self.publish_aruco()


    def publish_aruco(self):
        new = PoseArray()
        new.header.frame_id = "camera_color_optical_frame"
        new.header.stamp = rospy.Time.now()

        translate_x = []
        translate_y = []
        translate_z = []
        roll_x_lst = []
        pitch_y_lst = []
        yaw_z_lst = []

        if self.detect_flag == True:
            for i in range(len(self.corners)):
                new_pose = Pose()
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(self.corners[i], 0.06, self.camMatrix, self.distCoeffs)
                x, y, z = tvec[0][0][0], tvec[0][0][1], tvec[0][0][2]
                new_pose.position.x, new_pose.position.y, new_pose.position.z = x, y, z
                [[quat_x, quat_y, quat_z, quat_w]] = R.from_rotvec(rvec.reshape(1,3)).as_quat() ## R.from_{np.ndarray.shape (1,n)} > return type array([[_, _, _, _]]) (1, m)
                new_pose.orientation.x = quat_x
                new_pose.orientation.y = quat_y
                new_pose.orientation.z = quat_z
                new_pose.orientation.w = quat_w
                new.poses.append(new_pose)

                translate_x.append(round(x * 100, 2))
                translate_y.append(round(y * 100, 2))
                translate_z.append(round(z * 100, 2))
                [roll_x, pitch_y, yaw_z] = R.from_quat(np.array([quat_x, quat_y, quat_z, quat_w])).as_euler('xyz', degrees=True) ## input type (4,) > output type (3,)
                roll_x_lst.append(round(roll_x, 2))
                pitch_y_lst.append(round(pitch_y, 2))
                yaw_z_lst.append(round(yaw_z, 2))

                cv2.aruco.drawDetectedMarkers(self.frame, self.corners)
                cv2.drawFrameAxes(self.frame, self.camMatrix, self.distCoeffs, rvec, tvec, 0.03, 1)

            self.put_text([translate_x, translate_y, translate_z, roll_x_lst, pitch_y_lst, yaw_z_lst])

        else:
            new_pose = Pose()

        frame_send = self.bridge.cv2_to_imgmsg(self.frame, encoding="bgr8")
        self.pub_aruco_pose.publish(new)
        self.pub_aruco_frame_img.publish(frame_send)
        self.detect_flag = False

    def put_text(self, pose_lst):


        pose_lst = np.array(pose_lst).reshape(6, -1)


        for j in range(len(self.corners)):
            corner = self.corners[j].flatten()[0:2]

            cv2.putText(self.frame, f"{self.ids[j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]-60)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[0][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]-40)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[1][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]-20)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[2][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1])), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[3][j]}",(int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]+20)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[4][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]+40)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))
            cv2.putText(self.frame, f"{pose_lst[5][j]}", (int(corner[0]+50*(-1)**self.ids[j]), int(corner[1]+60)), cv2.FONT_ITALIC, 0.5, (255, 0, 0))


if __name__ == '__main__':
    rospy.init_node('realsense_aruco_pub')

    ad = Aruco_Detect()
    rate = rospy.Rate(190)

    while not rospy.is_shutdown():
        ad.find_aruco()
        # rate.sleep()
        print("_________________________________")