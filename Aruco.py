## https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/blob/main/pose_estimation.py

import cv2

def find_aruco(img):
    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    arucoParam = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParam)

    corners, ids, rejected = detector.detectMarkers(img)
    return corners, ids, rejected

def aruco_display(corners, ids, rejected, image):
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners
			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))


			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 8, (0, 0, 255), -1)

			cv2.circle(image, (topLeft[0], topLeft[1]), 10, (255, 0, 0), -1)
			cv2.circle(image, (topRight[0], topRight[1]), 10, (255, 0, 0), -1)
			cv2.circle(image, (bottomRight[0], bottomRight[1]), 10, (255, 0, 0), -1)
			cv2.circle(image, (bottomLeft[0], bottomLeft[1]), 10, (255, 0, 0), -1)

			cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
				3, (150, 150, 50), 5)
			print("[Inference] ArUco marker ID: {}".format(markerID))
			# show the output image
	return image


VideoCap = False
Video_filename = '~~~.avi'
cap = cv2.VideoCapture(Video_filename)

while True:
	if VideoCap:
		VideoCap, frame = cap.read()
		fps = cap.get(cv2.CAP_PROP_FPS)
		delay = round(100 / fps)
		print("FRAME FPS ; ", int(cap.get(cv2.CAP_PROP_FPS)))
	else:
		frame = cv2.imread("abc.jpg")
		frame2 = cv2.imread("def.jpg")




	corner_1, ids_1, rejected_1 = find_aruco(frame)
	# corner_2, ids_2, rejected_2 = find_aruco(frame2)

	detected_markers = aruco_display(corner_1, ids_1, rejected_1, frame)
	# detected_markers = aruco_display(corner_2, ids_2, rejected_2, frame2)

	cv2.namedWindow("image", flags=cv2.WINDOW_KEEPRATIO)
	cv2.imshow("image", detected_markers)


	cv2.waitKey(0)
	if cv2.waitKey() == 27:
		break
# cap.release()
cv2.destroyAllWindows()
