#!/usr/bin/env python2
import argparse
import rospy
import cv2
import sys, os
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Empty
from detection import detectionInterface
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


DEFAULT_RATE = 30 # [hz] Rate of publishing
DEFAULT_VIDEO_DEV = 0
DEFAULT_VISUALISATION = True
DEFAULT_METHOD = 'kfc'

NODE_NAME = 'obj_detection'
MAIN_TOPIC_NAME = 'obj_detection_topic'
VISUALISATION_TOPIC_NAME = 'obj_detection_vis'
SERVICE_NAME = 'obj_detection_calibration'

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rate', type=float, required=False, default=DEFAULT_RATE, help='rate of publishing [hz]')
    parser.add_argument('-d', '--device', type=int, required=False, default=DEFAULT_VIDEO_DEV, help='number of videostream source device')
    parser.add_argument('-v', '--visualisation', type=bool, required=False, default=DEFAULT_VISUALISATION, help='visualisation (true or false)')
    parser.add_argument('-m', '--method', type=str, required=False, default=DEFAULT_METHOD, help='method of tracking')
    args, unknown = parser.parse_known_args()
    return args.rate, args.device, args.method, args.visualisation

def talker():
    rate, device, method, visualisation = getArgs()

    # Init ros node, publisher and servivce
    rospy.init_node(NODE_NAME, anonymous=True)
    pub = rospy.Publisher(MAIN_TOPIC_NAME, PoseStamped, queue_size=1)
    if visualisation:
        img_pub = rospy.Publisher(VISUALISATION_TOPIC_NAME, Image, queue_size=1)
    calib_serv = rospy.Service(SERVICE_NAME, Empty, calibration)
    bridge = CvBridge()
    rate = rospy.Rate(rate)

    # Init detection object
    global di
    di = detectionInterface(device, method, visualisation)

    rospy.loginfo('Method: ', method)
    rospy.loginfo('Visualisation: ', visualisation)
    rospy.loginfo('Working...')

    while not rospy.is_shutdown():
        try:
            x, y = di.getObjectCoordinates()
            msg = PoseStamped()
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.header.stamp = rospy.Time.now()
            pub.publish(msg)
        except  Exception as e:
            rospy.logwarn("Exception: ")
            rospy.logwarn(str(e))

        # Publish image to image topic
        try:
            if visualisation:
                img_pub.publish(bridge.cv2_to_imgmsg(di.getImage(), "bgr8"))
        except  Exception as e:
            rospy.logwarn("Exception: ")
            rospy.logwarn(str(e))

        rate.sleep()

def calibration(req):
    try:
        di.calibration()
    except Exception as e:
        rospy.logwarn("Exception: ")
        rospy.logwarn(str(e))
    return {}

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass