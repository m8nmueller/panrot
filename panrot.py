import cv2
import numpy as np
import argparse
from functools import reduce

no_warp = np.array([[1, 0, 0], [0, 1, 0]])

def debug_points(img1, img2, points1, points2):
    cv2.imshow('Pre points', reduce(lambda im, p: cv2.circle(im, (int(p[1][0][0]), int(p[1][0][1])), 10, (p[0], 0, 200 - p[0]), -1), enumerate(points1), img1))
    cv2.imshow('Post points', reduce(lambda im, p: cv2.circle(im, (int(p[1][0][0]), int(p[1][0][1])), 10, (p[0], 0, 200 - p[0]), -1), enumerate(points2), img2))
    cv2.waitKey(0) # note that it waits for keypress inside image window

def debug_warps(imgs):
    warps = []
    for i in range(0, len(imgs) - 1):
        warps.append(getWarp(imgs[i], imgs[i+1]))

    shape = imgs[0].shape
    shape = (shape[1], shape[0])

    for (idx, img) in enumerate(reversed(imgs)):
        warp = no_warp
        idx = len(imgs) - 1 - idx
        for wind in range(idx, len(imgs) - 1):
            warp = multiplyAffine(warps[wind], warp)
            frame = cv2.warpAffine(img, warp, shape)
            cv2.imwrite(f'temp/img{idx}-warpto{wind+1}.jpg', frame)

# see https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
def getWarp(img1, img2):
    # Initiate SIFT detector
    sift = cv2.SIFT.create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>30:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        #debug_points(img1, img2, src_pts, dst_pts)
        warp_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        return warp_matrix
    else:
        print(f'not enough matches: {len(good)}')

# nice for affine stuff and show/plot basics: https://learnopencv.com/image-rotation-and-translation-using-opencv/
def renderWarp(img1, img2, warp1to2, out, dur, fps, forward):
    shape = img1.shape
    shape = (shape[1], shape[0])

    if forward:
        warp = warp1to2
    else:
        warp = cv2.invertAffineTransform(warp1to2)

    writer = cv2.VideoWriter(out, cv2.VideoWriter.fourcc(*'mp4v'), fps, shape)
    for alpha in np.linspace(0, 1, num=int(fps * dur)):
        if forward:
            frame = cv2.warpAffine(img1, (1-alpha) * no_warp + alpha * warp, shape)
        else:
            frame = cv2.warpAffine(img2, alpha * no_warp + (1-alpha) * warp, shape)
        writer.write(frame)
    writer.release()

def translateImages(first, second, out, dur, fps, forward):
    img1 = cv2.imread(first, cv2.IMREAD_COLOR)
    img2 = cv2.imread(second, cv2.IMREAD_COLOR)

    fullm = getWarp(img1, img2)
    renderWarp(img1, img2, fullm, out, dur, fps, forward)

# see https://stackoverflow.com/questions/40306194/combine-two-affine-transformations-matrices-in-opencv
def multiplyAffine(m1, m2):
    m1p = np.eye(3)
    m2p = np.eye(3)
    m1p[0:2, :] = m1
    m2p[0:2, :] = m2
    return (m1p @ m2p)[0:2, :]

def translateImagesWithOrig(steps, first, second, out, dur, fps, forward):
    img1 = cv2.imread(first, cv2.IMREAD_COLOR)
    img2 = cv2.imread(second, cv2.IMREAD_COLOR)
    imgs = [cv2.imread(step, cv2.IMREAD_COLOR) for step in steps]
    imgs.insert(0, img1)
    imgs.append(img2)

    warp = no_warp
    for i in range(0, len(imgs) - 1):
        warp = multiplyAffine(getWarp(imgs[i], imgs[i+1]), warp)

    renderWarp(img1, img2, warp, out, dur, fps, forward)



parser = argparse.ArgumentParser(description="A simple tool to create a video transforming one image to another by affine transformation")
 
# add arguments to the parser
parser.add_argument("-f", "--fps", default=24, type=int, help="FPS for the rendered video (default 24)")
parser.add_argument("-d", "--duration", default=2.0, type=float, help="time in seconds (float, default: 2)")
parser.add_argument("-o", "--out", default="out.mp4", help="the output file location (default out.mp4)")
parser.add_argument("first", help="the start image")
parser.add_argument("second", help="the end image")
parser.add_argument('--steps', nargs='*', help="supporting images in between (best use this after first/second)")
parser.add_argument('--forward', action='store_true', help="transform the first image to become the second (default: vice-versa)")

# parse the arguments
args = parser.parse_args()

if args.steps == None:
    print("translating without steps")
    translateImages(args.first, args.second, args.out, args.duration, args.fps, args.forward)
else:
    print(f"translating with steps {args.steps}")
    translateImagesWithOrig(args.steps, args.first, args.second, args.out, args.duration, args.fps, args.forward)
