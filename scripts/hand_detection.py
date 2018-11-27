import numpy as np
import cv2
import math
import random

BLUR = 5
DILATE_ITER = 1
ERODE_ITER = 1
random.seed()
deg2rad = math.pi / 180

def nothing(x):
    pass

def circularMean(array, lower_value, upper_value):
    print("mean arr", array)
    k = 360/(upper_value - lower_value)
    average_sin = 0
    average_cos = 0
    n = len(array)
    for i in range(n):
        average_sin += math.sin((array[i] * k - lower_value)*deg2rad)
        average_cos += math.cos((array[i] * k - lower_value)*deg2rad)
    average_sin = average_sin / n
    average_cos = average_cos / n
    correct_angle = lambda x: x if x >= 0 else 360 + x
    mean = correct_angle(math.atan2(average_sin, average_cos) / deg2rad)
    return int(mean/k)

def getRoi(image):
    h, w, _ = image.shape
    a = w / 9
    b = h / 6
    x1 = int(w / 2 - a)
    y1 = int(h / 2 - b)
    x2 = int(w / 2 + a)
    y2 = int(h / 2 + b)
    return x1, y1, x2, y2

def getPoints(image, amount_points):
    point_array =[]
    x1, y1, x2, y2 = getRoi(image)
    # for i in range(amount_points):
    #     x = random.randint(x1,x2)
    #     y = random.randint(y1,y2)
    #     point_array.append(image[x,y])
    width_p = int((x2 - x1 - 5)/amount_points)
    height_p = int((y2 - y1 - 5)/amount_points)
    for i in range(amount_points):
        for j in range(amount_points):
            x = x1 + 5 + width_p*i
            y = y1 + 5 + height_p*j
            point_array.append(image[x,y])
    return np.array(point_array)

def getConstraints(image):
    n = 5
    hue_margin = 12
    sat_margin = 10
    value_margin = 10
    points = getPoints(image, n)
    hue_array = []
    sat_array = []
    value_array = []
    for i in range(len(points)):
        hue_array.append(points[i,0])
        sat_array.append(points[i,1])
        value_array.append(points[i,2])
    mean = circularMean(hue_array, 0, 180)
    check_hue = lambda x, low, up: x if (x>=low and x<=up) else x-np.sign(x)*up
    lower = np.array([check_hue(mean - hue_margin,0,180), min(sat_array)-sat_margin, min(value_array)-value_margin])
    upper = np.array([check_hue(mean + hue_margin,0,180), max(sat_array)+sat_margin, max(value_array)+value_margin])
    return lower, upper

def getMask(image, lower, upper):
    if lower[0] <= upper[0]:
        mask = cv2.inRange(image, lower, upper)
    else:
        lower1 = np.array([0, lower[1], lower[2]])
        upper1 = np.array([upper[0], upper[1], upper[2]])
        lower2 = np.array([lower[0], lower[1], lower[2]])
        upper2 = np.array([255, upper[1], upper[2]])

        mask1 = cv2.inRange(image, lower1, upper1)
        mask2 = cv2.inRange(image, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    return mask

def centroid(contour):
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return cx, cy

def getObjectCoordinates(bin_image):
    kernel = np.ones((7,7),np.uint8)
    bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
    # bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_OPEN, kernel)
    bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)
    bin_image = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)
    bin_image = cv2.dilate(bin_image, kernel, iterations=DILATE_ITER)
    cv2.imshow("morph", bin_image)
    contours, hierarchy = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_cnt = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    return max_cnt



def mouseCallback(event, x, y, flags, param):
    global hsv
    global lower
    global upper

    pixel = [x, y]

    if event == cv2.EVENT_LBUTTONDBLCLK:
        lower, upper = getConstraints(hsv)
        print("new lower and upper")
        print(lower)
        print(upper)
        print()
    elif event == cv2.EVENT_LBUTTONUP:
        print(hsv[pixel[1], pixel[0]])



cam = cv2.VideoCapture(0)

cv2.namedWindow('hsv')
lower = np.array([0, 0, 0])
upper = np.array([255, 255, 255])
cv2.setMouseCallback('hsv', mouseCallback)

while True:
    _, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.medianBlur(hsv, BLUR)
    mask = getMask(hsv, lower, upper)
    x1, y1, x2, y2 = getRoi(frame)
    cv2.rectangle(hsv,(x1,y1),(x2, y2),(0,255,0),3)

    cnt = getObjectCoordinates(mask)
    if id(cnt) != id(0):
        cv2.drawContours(frame, cnt, -1, (0, 255, 0),2)
        cx, cy = centroid(cnt)
        cv2.circle(frame,(cx,cy), 4, (255,0,255), 2)

    cv2.imshow("mask", mask)
    cv2.imshow("hsv", hsv)
    cv2.imshow("frame", frame)
    # cv2.imshow("morph", bin_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()