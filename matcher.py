import cv2 as cv
import numpy as np

class Matcher:
    def __init__(self):
        pass
    
    def knnMatch(self, desA, desB, ratio):
        # setup the OpenCV descriptor matcher
        matcher = cv.BFMatcher()

        # use the knnMatch to match descriptors
        matches = matcher.knnMatch(desA, desB, k=2)

        # initialize the list of actual matches
        matches = [(m[0].trainIdx, m[0].queryIdx) for m in matches if len(m) == 2 and m[0].distance < m[1].distance * ratio]

        return matches