import cv2 as cv
import numpy as np
import sys, os, math, argparse
from matplotlib import pyplot as plt

def main():
    print(cv.__version__)

    # read an image
    img = cv.imread('pics/topic.png', cv.IMREAD_COLOR)
    print(img.shape) # (617, 802, 3) 
    print(type(img)) # <class 'numpy.ndarray'>

    # show an image
    cv.imshow('image', img)
    cv.waitKey(0) # wait for a key press
    cv.destroyAllWindows()

    # write an image
    cv.imwrite('pics/topic_copy.png', img)

if __name__ == '__main__':
    main()