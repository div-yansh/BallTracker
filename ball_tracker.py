import time
import imutils
import cv2 as cv
import numpy as np
from collections import deque
from argparse import ArgumentParser
from imutils.video import VideoStream


# initialise argument parser
ap = ArgumentParser()
ap.add_argument("--video", help="path to video")
ap.add_argument("--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# set lower and upper bounds for green color
lower_bound = (29, 86, 6)
upper_bound = (64, 255, 255)

# set tail length
pts = deque(maxlen = args["buffer"])

# if a video is provided, set reference to it, else to webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
else:
	vs = cv.VideoCapture(args["video"])

while True:
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# jump out if video ends
	if frame is None:
		break

	# resize image, blur it, and convert to HSV
	frame = imutils.resize(frame, width=600)
	blur = cv.GaussianBlur(frame, (11, 11), 0)
	hsv_frame = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
	
	# construct a maks for green color 
	# then perform dilation and erosion to remove unwabted mask
	mask = cv.inRange(hsv_frame, lower_bound, upper_bound)
	mask = cv.erode(mask, None, iterations=2)
	mask = cv.dilate(mask, None, iterations=2)

	# find contour of ball i.e. masl
	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, 
						   cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	centre = None

	# a ball must be having atleast 1 contour
	if len(cnts) >= 1:
		# find the largest contour, draw approx circle and find centroid
		c = max(cnts, key=cv.contourArea)
		(x,y), radius = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		centre = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

		# only proceed if radius is greater than minimum size
		if radius > 10:
			# draw circle and centroid
			cv.circle(frame, (int(x),int(y)), int(radius), (0,255,0), 2)
			cv.circle(frame, centre, 5, (0,0,255), -1)

		# update deque with centroid value
		pts.appendleft(centre)

	# draw ball tail
	for i in range(1, len(pts)):
		# if any point is None, ignore it
		if pts[i-1] is None or pts[i] is None:
			continue

		# draw tail
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

	# show video
	cv.imshow("mask", frame)
	if cv.waitKey(1) & 0xFF == 27:
		break

if not args.get("video", False):
	vs.stop()
else:
	vs.release()
cv.destroyAllWindows()