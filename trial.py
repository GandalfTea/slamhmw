#!usr/bim/python3

#debug
import time

import cv2
import numpy as np
import sys
from skimage.measure import ransac 
from skimage.transform import FundamentalMatrixTransform

# video split
W = 1280
H = 720 

# 254, 240
GW = 254
GH = 240

if len(sys.argv) < 2:
    print("Error: Video argument not found.\n")
    sys.exit()

orb = cv2.ORB_create()
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
vid = cv2.VideoCapture(sys.argv[1])

if (vid.isOpened() == False):
    print("Error opening video.");
    sys.exit()


class Frame(object):
    def __init__(self, img, kp, des):
        self.img = img
        self.kp = kp
        self.des = des


frames = [] 
while(vid.isOpened()):
    start = time.time()
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(frames) > 2: frames.pop(0)

    if ret == True:
        kps= []
        for rw in range(0, W, GW):
            for rh in range(0, H, GH):
                a = gray[rh:rh+GH, rw:rw+GW]
                pts = cv2.goodFeaturesToTrack(a, 3000, qualityLevel=0.05, minDistance=7)

                if pts is not None:
                    # recalibrating data 
                    kp = [cv2.KeyPoint(x=f[0][0] + rw, y=f[0][1] + rh, size=20) for f in pts]
                    kps.extend(kp)
        kp, des = sift.compute(gray, kps)
        frames.append(Frame(gray, kps, des))

        if len(frames) > 1:
            # matches
            f2 = frames[-2]

            matches = bf.knnMatch(des, np.array(f2.des), k=2)

            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    if (m.distance < 32):
                        good.append(( kps[m.queryIdx].pt,  f2.kp[m.trainIdx].pt ))

            assert(len(set(good)) == len(good))

            if len(good) > 8:
                good = np.array(good)
                #print(good.shape)
                
                model, inliers = ransac((good[:, 0], good[:, 1]),
                                        FundamentalMatrixTransform,
                                        min_samples=8,
                                        residual_threshold=1,
                                        max_trials=100)
                good = good[inliers]
                #print(sum(inliers))

            for (kp1, kp2) in good:
                kp1s = list(map(lambda x: int(round(x)), kp1))
                kp2s = list(map(lambda x: int(round(x)), kp2))
                cv2.line(frame, kp1s, kp2s, (255,0,0), 2)

        for p in kps:
            x, y = int(p.pt[0]), int(p.pt[1]) 
            cv2.circle(frame, (x, y),1,(0,255,0), 1)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    stop = time.time()
    print(stop-start)

vid.release()
cv2.destroyAllWindows()

