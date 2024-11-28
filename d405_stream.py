import pyrealsense2 as rs
import numpy as np
import cv2

ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()
print('Reset : done')
# Camera streams
res_color = (848, 480)
res_depth = (848, 480)
fps = 30


# Colorizer for the depth frame
# colorizer = rs.colorizer()

# Initialize D405
pipe_D405 = rs.pipeline()
config_D405 = rs.config()
config_D405.enable_stream(rs.stream.depth, res_color[0], res_color[1], rs.format.z16, fps)
config_D405.enable_stream(rs.stream.color, res_depth[0], res_depth[1], rs.format.bgr8, fps)
align_to = rs.stream.color
align = rs.align(align_to)

# Start pipeline
prof_D405 = pipe_D405.start(config_D405)

# d405_distCoeffs = np.array([-0.05458524078130722, 0.057370759546756744, 0.00011702199117280543, 0.0012725829146802425, -0.018503589555621147])
# d405_camMatrix = np.array([[430.583740234375, 0.0, 419.8208312988281], [0.0, 430.1481628417969, 239.1549835205078], [0.0, 0.0, 1.0]])

def stream():
    while True:
        frames_D405 = pipe_D405.wait_for_frames()
        depth = frames_D405.get_depth_frame()

        depth_data = depth.as_frame().get_data()
        depth_image = np.asanyarray(depth_data)
        # alignment
        frames_D405 = align.process(frames_D405)

        # Get color and depth frames for display
        color_frame = frames_D405.get_color_frame()
        color_frame = np.asanyarray(color_frame.get_data())

        # Display images
        cv2.imshow("RealSense", color_frame)
        # cv2.imshow("depth", depth_image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            cv2.destroyWindow("RealSense")
            pipe_D405.stop()
            break

        if key == ord('c'):     # capture
            cv2.imwrite("d405_capture_occlude2.jpg", color_frame)
            print("capture done")
            quit()



if __name__ == "__main__":
    stream()