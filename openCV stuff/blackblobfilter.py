import cv2
import numpy
import math
from enum import Enum

class DoubleBlackBlobFilter:
    
    def __init__(self):
        self.cv_add_output = None
        self.__blur_input = self.cv_add_output
        self.__blur_type = BlurType.Gaussian_Blur
        self.__blur_radius = 5.405405405405405
        self.blur_output = None
        self.__hsv_threshold_input = self.blur_output
        self.__hsv_threshold_hue = [61.510791366906474, 121.63822525597271]
        self.__hsv_threshold_saturation = [64.20863309352518, 239.76962457337885]
        self.__hsv_threshold_value = [0.0, 96.1689419795222]
        self.hsv_threshold_output = None
        self.__find_blobs_input = self.hsv_threshold_output
        self.__find_blobs_min_area = 1
        self.__find_blobs_circularity = [0.0, 1.0]
        self.__find_blobs_dark_blobs = False
        self.find_blobs_output = None


    def process(self, source0):
        self.__cv_add_src1 = source0
        self.__cv_add_src2 = source0
        (self.cv_add_output) = self.__cv_add(self.__cv_add_src1, self.__cv_add_src2)

        self.__blur_input = self.cv_add_output
        (self.blur_output) = self.__blur(self.__blur_input, self.__blur_type, self.__blur_radius)

        self.__hsv_threshold_input = self.blur_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        self.__find_blobs_input = self.hsv_threshold_output
        (self.find_blobs_output) = self.__find_blobs(self.__find_blobs_input, self.__find_blobs_min_area, self.__find_blobs_circularity, self.__find_blobs_dark_blobs)


    @staticmethod
    def __cv_add(src1, src2):
        return cv2.add(src1,src2)

    @staticmethod
    def __blur(src, type, radius):
        if(type is BlurType.Box_Blur):
            ksize = int(2 * round(radius) + 1)
            return cv2.blur(src, (ksize, ksize))
        elif(type is BlurType.Gaussian_Blur):
            ksize = int(6 * round(radius) + 1)
            return cv2.GaussianBlur(src, (ksize, ksize), round(radius))
        elif(type is BlurType.Median_Filter):
            ksize = int(2 * round(radius) + 1)
            return cv2.medianBlur(src, ksize)
        else:
            return cv2.bilateralFilter(src, -1, round(radius), round(radius))

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __find_blobs(input, min_area, circularity, dark_blobs):
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = 1
        params.blobColor = (0 if dark_blobs else 255)
        params.minThreshold = 10
        params.maxThreshold = 220
        params.filterByArea = True
        params.minArea = min_area
        params.filterByCircularity = True
        params.minCircularity = circularity[0]
        params.maxCircularity = circularity[1]
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        return detector.detect(input)


BlurType = Enum('BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')

