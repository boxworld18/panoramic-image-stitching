import cv2 as cv
import numpy as np
import sys, os, math, argparse
from matplotlib import pyplot as plt
from Stitcher import Stitcher
from utils import cv_read, cv_show, cv_write
from define import PIC_1, PIC_2, PIC_OUT, PIC_CV_RESULT
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pic1', type=str, default=PIC_1, help='path to the first image')
    parser.add_argument('--pic2', type=str, default=PIC_2, help='path to the second image')
    parser.add_argument('--pic_out', type=str, default=PIC_OUT, help='path to the output image')
    parser.add_argument('--pic_cv_result', type=str, default=PIC_CV_RESULT, help='path to the output image')

    parser.add_argument('--ratio', type=float, default=0.75, help='ratio of keypoints')
    parser.add_argument('--reprojThresh', type=float, default=4.0, help='reprojection threshold')
    parser.add_argument('--fusionMethod', type=str, default="default", help='method of fusion')
    parser.add_argument('--showMatches', default=False, action='store_true', help='show matches or not')
    parser.add_argument('--NotShowAny', default=False, action='store_true', help='do not show any image')
    return parser.parse_args()
    
def main():
    args = parse_args()

    # read images
    imageA = cv_read(args.pic1)
    imageB = cv_read(args.pic2)

    # use my stitcher to stitch images
    stitcher = Stitcher()
    (result, vis) = stitcher.stitch(images=(imageA, imageB), ratio=args.ratio, 
                                    reprojThresh=args.reprojThresh, fusionMethod=args.fusionMethod, 
                                    showMatches=args.showMatches, showAny=(not args.NotShowAny))

    # show the result
    if not args.NotShowAny:
        cv_show("Result", result)
    cv_write(args.pic_out, result)

    # use OpenCV stitcher to stitch images
    stitcher = cv.Stitcher_create()
    (status, pano) = stitcher.stitch([imageA, imageB])
    print('OpenCV Stitcher Status: {}'.format(status))
    if status == cv.Stitcher_OK:
        if not args.NotShowAny:
            cv_show("OpenCV Stitcher", pano)
        cv_write(args.pic_cv_result, pano)
    
if __name__ == '__main__':
    main()