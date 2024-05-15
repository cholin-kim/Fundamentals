import rospy
import numpy as np
from geometry_msgs.msg import Pose, PoseArray

rospy.init_node("Aruco_random_publish")
aruco_pose_topic = "/aruco/pose"
pub_aruco_pose = rospy.Publisher(aruco_pose_topic, PoseArray, queue_size=1)

'''
1. execute roscore
2. run Aruco_random_publish.py
    Here, random poses are published through "/aruco/pose" msg
    randomly, 1 ~ 3 poses are published
3. rostopic echo {msg name}(new terminal)
4. use rostopic info {msg name} to check publisher & subscriber
'''

def create_random_pose(num_of_markers):
    msg = PoseArray()
    msg.header.frame_id = "camera_color_optical_frame"

    for n in range(num_of_markers):

        msg_pose = Pose()
        rand_lst = np.random.rand(7,)

        msg_pose.position.x = rand_lst[0]
        msg_pose.position.y = rand_lst[1]
        msg_pose.position.z = rand_lst[2]
        msg_pose.orientation.x = rand_lst[3]
        msg_pose.orientation.y = rand_lst[4]
        msg_pose.orientation.z = rand_lst[5]
        msg_pose.orientation.w = rand_lst[6]
        msg.poses.append(msg_pose)
    return msg


rate = rospy.Rate(5)
while True:
    num = np.random.randint(1, 4)
    print("num of aruco markers:", num)
    msg_pub = create_random_pose(num)
    pub_aruco_pose.publish(msg_pub)
    rate.sleep()