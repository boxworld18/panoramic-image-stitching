import cv2 as cv
import numpy as np
import sys, os, math, argparse
from matplotlib import pyplot as plt
from Stitcher import Stitcher
from utils import cv_read, cv_show, cv_write
from define import PIC_1, PIC_2, PIC_OUT, PIC_CV_RESULT
    
def main():
    # read images
    imageA = cv_read(PIC_1)
    imageB = cv_read(PIC_2)

    # use my stitcher to stitch images
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch(images=(imageA, imageB), showMatches=True)

    # show the result
    cv_show("Result", result)
    cv_write(PIC_OUT, result)

    # use OpenCV stitcher to stitch images
    stitcher = cv.Stitcher_create()
    (status, pano) = stitcher.stitch([imageA, imageB])
    print('OpenCV Stitcher Status: {}'.format(status))
    if status == cv.Stitcher_OK:
        cv_show("OpenCV Stitcher", pano)
        cv_write(PIC_CV_RESULT, pano)
    
if __name__ == '__main__':
    main()