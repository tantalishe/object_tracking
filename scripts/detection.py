import numpy as np
import cv2
import math

class detectionInterface:

	def __init__(self, video, method, visualisation):

		# Init video capture
		self.cam = cv2.VideoCapture(video)
		self.visualisation = visualisation
		# method can be
		# - color
		# - kfc
		self.method = method

		if self.method == 'kfc':
			self.tracker = cv2.TrackerKCF_create()
			self.cailib_flag = False
			self.bbox = (0,0,0,0)

		if self.method =='color':
			self.blur = 5
			self.dilate_iter = 1
			self.erode_iter = 1
			self.deg2rad = math.pi / 180

			# getConstrains variables
			self.n = 5
			self.hue_margin = 12
			self.sat_margin = 10
			self.value_margin = 10

			# thresholds
			self.lower = np.array([0, 0, 0])
			self.upper = np.array([255, 255, 255])

		# images
		self.frame = []
		self.hsv = []
		self.mask = []

	def getObjectCoordinates(self):
		_, self.frame = self.cam.read()

		cx = 0
		cy = 0

		if self.method == 'color':
			cx, cy = self.getObjectCoordinatesColor()

		if self.method == 'kfc':
			cx, cy = self.getObjectCoordinatesKFC()

		height, width, _ = self.frame.shape
		x = cx - int(width/2)
		y = cy - int(height/2)
		return x, y

	def getObjectCoordinatesColor(self):
		self.hsv = cv2.medianBlur(self.frame, self.blur)
		self.hsv = cv2.cvtColor(self.hsv, cv2.COLOR_BGR2HSV)
		self.mask = self.getMask(self.hsv, self.lower, self.upper)
		cnt, area = self.getMaxContour(self.mask)
		if (id(cnt) == id(0)):
			raise Exception("There are no any contours")
		if (area < 500) :
			raise Exception("Biggest contour too small")

		cx, cy = self.centroid(cnt)
		if self.visualisation:
			x1, y1, x2, y2 = self.getRoi(self.frame)
			cv2.rectangle(self.frame,(x1,y1),(x2, y2),(255,255,0),1)
			cv2.drawContours(self.frame, cnt, -1, (0, 255, 0),2)
			cv2.circle(self.frame,(cx,cy), 4, (255,0,255), 2)

		return cx, cy

	def getObjectCoordinatesKFC(self):
		if not self.cailib_flag:
			self.calibrationKFC()
			self.cailib_flag = True
		(success, self.bbox) = self.tracker.update(self.frame)
		if not success:
			raise Exception("Tracking failure")
		(x, y, w, h) = self.bbox
		cx = int(x + w/2)
		cy = int(y + h/2)
		if self.visualisation:
			cv2.rectangle(self.frame,(int(x),int(y)),(int(x+w), int(y+h)),(0,255,0),2)
			cv2.circle(self.frame,(cx,cy), 4, (255,0,255), 2)

		return cx, cy

	def getImage(self):
		x1, y1, x2, y2 = self.getRoi(self.frame)
		cv2.rectangle(self.frame,(x1,y1),(x2, y2),(255,255,0),1)
		return self.frame

	def calibration(self):
		if self.method == 'color':
			self.calibrationColor()
		if self.method == 'kfc':
			self.calibrationKFC()

	def calibrationColor(self):
		self.lower, self.upper = self.getConstrains(self.hsv)

	# Reinit tracker with new box
	def calibrationKFC(self):
		x1, y1, x2, y2 = self.getRoi(self.frame)
		self.bbox = (x1, y1, x2 - x1, y2 - y1)
		self.tracker = cv2.TrackerKCF_create()
		self.tracker.init(self.frame, self.bbox)
		print(self.tracker)

	# Return square in middle part of image
	def getRoi(self, image):
		height, width, _ = image.shape
		a = width / 8
		# b = h / 6
		x1 = int(width / 2 - a)
		y1 = int(height / 2 - a)
		x2 = int(width / 2 + a)
		y2 = int(height / 2 + a)
		return x1, y1, x2, y2

	# Part of Color method.
	# Return contour with max area
	def getMaxContour(self, bin_image):
		kernel = np.ones((7,7),np.uint8)
		bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
		# bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
		bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)
		bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)
		bin_image = cv2.dilate(bin_image, kernel, iterations=self.dilate_iter)
		cv2.imshow("morph", bin_image)
		contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		max_area = 0
		max_cnt = 0
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > max_area:
				max_area = area
				max_cnt = cnt
		return max_cnt, max_area

	# Part of Color method.
	# Smart inRange
	def getMask(self, image, lower, upper):
		if self.lower[0] <= self.upper[0]:
			mask = cv2.inRange(image, self.lower, self.upper)
		else:
			lower1 = np.array([0, self.lower[1], self.lower[2]])
			upper1 = np.array([self.upper[0], self.upper[1], self.upper[2]])
			lower2 = np.array([self.lower[0], self.lower[1], self.lower[2]])
			upper2 = np.array([255, self.upper[1], self.upper[2]])

			mask1 = cv2.inRange(image, lower1, upper1)
			mask2 = cv2.inRange(image, lower2, upper2)
			mask = cv2.bitwise_or(mask1, mask2)

		return mask

	# Part of Color method
	def centroid(self, contour):
		M = cv2.moments(contour)
		cx = int(M['m10']/M['m00'])
		cy = int(M['m01']/M['m00'])
		return cx, cy

	# Part of Color method.
	# Return mean for some circular quantities (like hue in HSV)
	def circularMean(self, array, lower_value, upper_value):
		k = 360/(upper_value - lower_value)
		average_sin = 0
		average_cos = 0
		n = len(array)
		for i in range(n):
			average_sin += math.sin((array[i] * k - lower_value)*self.deg2rad)
			average_cos += math.cos((array[i] * k - lower_value)*self.deg2rad)
		average_sin = average_sin / n
		average_cos = average_cos / n
		correct_angle = lambda x: x if x >= 0 else 360 + x
		mean = correct_angle(math.atan2(average_sin, average_cos) / self.deg2rad)
		return int(mean/k)

	# Part of Color method.
	# Return points from roi
	def getPoints(self, image, amount_points):
		point_array =[]
		x1, y1, x2, y2 = self.getRoi(image)
		width_p = int((x2 - x1 - 5)/amount_points)
		height_p = int((y2 - y1 - 5)/amount_points)
		for i in range(amount_points):
			for j in range(amount_points):
				x = x1 + 5 + width_p*i
				y = y1 + 5 + height_p*j
				point_array.append(image[x,y])
		return np.array(point_array)

	# Part of Color method.
	# Return new lower and upper for inRange function
	def getConstrains(self,image):
		points = self.getPoints(image, self.n)
		hue_array = []
		sat_array = []
		value_array = []
		for i in range(len(points)):
			hue_array.append(points[i,0])
			sat_array.append(points[i,1])
			value_array.append(points[i,2])
		mean = self.circularMean(hue_array, 0, 180)
		check_hue = lambda x, low, up: x if (x>=low and x<=up) else x-np.sign(x)*up
		lower = np.array([check_hue(mean - self.hue_margin,0,180), min(sat_array)-self.sat_margin, min(value_array)-self.value_margin])
		upper = np.array([check_hue(mean + self.hue_margin,0,180), max(sat_array)+self.sat_margin, max(value_array)+self.value_margin])
		return lower, upper