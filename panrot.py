import cv2
import numpy as np
import argparse
#import sys
#from affineTransformTools import getTranslationX, getTranslationY, getRotation, getAffineTransform

no_warp = np.array([[1, 0, 0], [0, 1, 0]])

def getWarp0(img1, img2, prev_points: cv2.UMat):
    points, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, prev_points, None)
    indices = np.where(status==1)[0]
    warp_matrix, _ = cv2.estimateAffinePartial2D(prev_points[indices], points[indices], method=cv2.LMEDS)
    return warp_matrix

# thanks to https://forum.opencv.org/t/measure-pan-rotate-and-resize-between-frames/16149/2 for algo
# see https://stackoverflow.com/questions/54483794/what-is-inside-how-to-decompose-an-affine-warp-matrix for warp matrix decompose

def getWarp(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(img1_gray,maxCorners=200,qualityLevel=0.01,minDistance=30,blockSize=3)
    return getWarp0(img1, img2, prev_points)

def renderWarp(img1, img2, warp1to2, out, dur, fps, forward):
    shape = img1.shape
    shape = (shape[1], shape[0])

    if forward:
        warp = warp1to2
    else:
        warp = cv2.invertAffineTransform(warp1to2)

    writer = cv2.VideoWriter(out, cv2.VideoWriter.fourcc(*'mp4v'), fps, shape)
    for alpha in np.linspace(0, 1, num=fps * dur):
        if forward:
            frame = cv2.warpAffine(img1, (1-alpha) * no_warp + alpha * warp, shape)
        else:
            frame = cv2.warpAffine(img2, alpha * no_warp + (1-alpha) * warp, shape)
        writer.write(frame)
    writer.release()

def translateImages(first: str, second: str, out: str, dur: int, fps: int, forward):
    img1 = cv2.imread(first, cv2.IMREAD_COLOR)
    img2 = cv2.imread(second, cv2.IMREAD_COLOR)

    fullm = getWarp(img1, img2)
    renderWarp(img1, img2, fullm, out, dur, fps, forward)

def multiplyAffine(m1, m2):
    m1p = np.zeros((3,3))
    m2p = np.zeros((3,3))
    m1p[0:2, :] = m1
    m2p[0:2, :] = m2
    return (m1p * m2p)[0:2, :]
    # res = np.empty((2, 3))
    # res[0:2, 0:2] = m1[0:2, 0:2] * m2[0:2, 0:2]
    # res[:, 2] = m1[:, 2] + m2[:, 2]
    # return res

def translateImagesWithOrig(steps, first: str, second: str, out: str, dur: int, fps: int, forward):
    img1 = cv2.imread(first, cv2.IMREAD_COLOR)
    img2 = cv2.imread(second, cv2.IMREAD_COLOR)
    imgs = [cv2.imread(step, cv2.IMREAD_COLOR) for step in steps]
    imgs.insert(0, img1)
    imgs.append(img2)

    warp = no_warp
    for i in range(0, len(imgs) - 1):
        warp = multiplyAffine(getWarp(imgs[i], imgs[i+1]), warp)

    renderWarp(img1, img2, warp, out, dur, fps, forward)



parser = argparse.ArgumentParser()
 
# add arguments to the parser
parser.add_argument("-f", "--fps", default=24, type=int)
parser.add_argument("-d", "--dur", default=2, type=int)
parser.add_argument("-o", "--out", default="out.mp4")
parser.add_argument("first")
parser.add_argument("second")
parser.add_argument('--step', nargs='*')
parser.add_argument('--forward', action='store_true')

# parse the arguments
args = parser.parse_args()

if args.step == None:
    print("translating without steps")
    translateImages(args.first, args.second, args.out, args.dur, args.fps, args.forward)
else:
    print(f"translating with steps {args.step}")
    translateImagesWithOrig(args.step, args.first, args.second, args.out, args.dur, args.fps, args.forward)
