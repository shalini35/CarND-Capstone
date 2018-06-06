from styx_msgs.msg import TrafficLight

#importing some useful packages

#import matplotlib.pyplot as plt

#import matplotlib.image as mpimg

import numpy as np

import cv2

import math

#%matplotlib inline



class TLClassifier(object):

    def __init__(self):

        #TODO load classifier

        pass



    def get_classification(self, image):

        """Determines the color of the traffic light in the image

        Args:

            image (cv::Mat): image containing the traffic light

        Returns:

            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

        # red has hue 0 - 10 & 160 - 180 add another filter 

        # TODO  use Guassian mask

        R_min1 = np.array([0, 100, 100],np.uint8)

        R_max1 = np.array([10, 255, 255],np.uint8)        



        R_min2 = np.array([160, 100, 100],np.uint8)

        R_max2 = np.array([179, 255, 255],np.uint8)



        threshed1 = cv2.inRange(hsv_image, R_min1, R_max1) 

        threshed2 = cv2.inRange(hsv_image, R_min2, R_max2) 

        if cv2.countNonZero(threshed1) + cv2.countNonZero(threshed2) > 47:

            return TrafficLight.RED



        Y_min = np.array([40.0/360*255, 100, 100],np.uint8)

        Y_max = np.array([66.0/360*255, 255, 255],np.uint8)

        threshed3 = cv2.inRange(hsv_image, Y_min, Y_max)

        if cv2.countNonZero(threshed3) > 47:

            return TrafficLight.YELLOW



        G_min = np.array([90.0/360*255, 100, 100],np.uint8)

        G_max = np.array([140.0/360*255, 255, 255],np.uint8)

        threshed4 = cv2.inRange(hsv_image, G_min, G_max)

        if cv2.countNonZero(threshed4) > 47:

            return TrafficLight.GREEN





        return TrafficLight.UNKNOWN