import cv2
import numpy as np
import argparse
#import sys
#from affineTransformTools import getTranslationX, getTranslationY, getRotation, getAffineTransform

no_warp = np.matrix([[1, 0, 0], [0, 1, 0]])

def getWarp(img1, img2, prev_points: cv2.UMat):
    points, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, prev_points, None)
    indices = np.where(status==1)[0]
    warp_matrix, _ = cv2.estimateAffinePartial2D(prev_points[indices], points[indices], method=cv2.LMEDS)
    return warp_matrix

# thanks to https://forum.opencv.org/t/measure-pan-rotate-and-resize-between-frames/16149/2 for algo
# see https://stackoverflow.com/questions/54483794/what-is-inside-how-to-decompose-an-affine-warp-matrix for warp matrix decompose

def translateImages(first: str, second: str, out: str, dur: int, fps: int):
    img1 = cv2.imread(first, cv2.IMREAD_COLOR)
    img2 = cv2.imread(second, cv2.IMREAD_COLOR)

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    prev_points = cv2.goodFeaturesToTrack(img1_gray,maxCorners=200,qualityLevel=0.01,minDistance=30,blockSize=3)
    fullm = getWarp(img1_gray, img2_gray, prev_points)
    fullm = cv2.invertAffineTransform(fullm)

    shape = img1.shape
    shape = (shape[1], shape[0])

    print(f"shape is {shape[0]}  ,  {shape[1]}")

    # dx = getTranslationX(warp_matrix)
    # dy = getTranslationY(warp_matrix)
    # df = getRotation(warp_matrix)

    #print(f"translation x: {dx} y: {dy}, and rotation {df}")

    writer = cv2.VideoWriter(out, cv2.VideoWriter.fourcc(*'mp4v'), fps, shape)
    for alpha in np.linspace(0, 1, num=fps * dur):
        frame = cv2.warpAffine(img2, alpha * no_warp + (1-alpha) * fullm, shape)
        writer.write(frame)
    writer.release()

def multiplyAffine(m1, m2):
    res = np.empty((2, 3))
    res[0:1, 0:1] = m1[0:1, 0:1] * m2[0:1, 0:1]
    res[:, 2] = m1[:, 2] + m2[:, 2]
    return res

def translateImagesWithOrig(orig: str, first: str, second: str, out: str, dur: int, fps: int):
    img0 = cv2.imread(orig, cv2.IMREAD_COLOR)
    img1 = cv2.imread(first, cv2.IMREAD_COLOR)
    img2 = cv2.imread(second, cv2.IMREAD_COLOR)

    shape = img1.shape
    shape = (shape[1], shape[0])

    img0_gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    prev_points = cv2.goodFeaturesToTrack(img0_gray,maxCorners=200,qualityLevel=0.01,minDistance=30,blockSize=3)
    m1 = getWarp(img0_gray, img1_gray, prev_points)
    m2 = getWarp(img0_gray, img2_gray, prev_points)


    # cv2.imshow('Original image 1', img0)
    # cv2.imshow('Original image 1', img1)
    # cv2.imshow('Warped', cv2.warpAffine(img0, m1, shape))
    # # wait indefinitely, press any key on keyboard to exit
    # cv2.waitKey(0)
    # return



    writer = cv2.VideoWriter(out, cv2.VideoWriter.fourcc(*'mp4v'), fps, shape)
    for alpha in np.linspace(0, 1, num=fps * dur):
        frame = cv2.warpAffine(img0, ((1-alpha) * m1) + (alpha * m2), shape)
        writer.write(frame)
    writer.release()



parser = argparse.ArgumentParser()
 
# add arguments to the parser
parser.add_argument("-f", "--fps", default=24, type=int)
parser.add_argument("-d", "--dur", default=2, type=int)
parser.add_argument("-o", "--out", default="out.mp4")
parser.add_argument("first")
parser.add_argument("second")
parser.add_argument('--origin')

# parse the arguments
args = parser.parse_args()

if args.origin == None:
    print("translating without origin")
    translateImages(args.first, args.second, args.out, args.dur, args.fps)
else:
    print(f"translating with origin {args.origin}")
    translateImagesWithOrig(args.origin, args.first, args.second, args.out, args.dur, args.fps)
