#!usr/bim/python3

#debug
import time

import cv2
import numpy as np
import sys
from skimage.measure import ransac 
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

np.set_printoptions(suppress=True)

# video split
W = 1280 
H = 720 

# 254, 240
GW = 112 
GH = 126

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

# turn [x, y] -> [x, y, 1]
def pad(p):
    return np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)

class Frame(object):
    def __init__(self, img, kp, des, F, K, Kinv):
        self.img = img
        self.kp = kp
        self.des = des
        self.F = F
        self.K = K 
        self.Kinv = Kinv 


frames = [] 
while(vid.isOpened()):
    start_full = time.time()

    ret, frame = vid.read()
    frame = cv2.resize(frame, (896, 504), interpolation = cv2.INTER_AREA)
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

        _, des = sift.compute(gray, kps)

        # by model values, it should be 100
        F = 450
        K = np.array([[F, 0, gray.shape[1]//2],
                      [0, F, gray.shape[0]//2],
                      [0, 0, 1]])
        Kinv = np.linalg.inv(K)

        frames.append(Frame(gray, kps, des, F, K, Kinv))


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

                # Normalized coordinates are independent of the resolution of the image and give better 
                # numerical stability for some multi-view geometry algorithms than pixel coordinates.
                # Normalized points are stored in values -1 - 1, with 1 being the larger side of the picture
                # The origin is the middle of the image. 
                # first one is changing the queryIdx, second trainIdx

                # order of change is queryIdx followed by trainIdx

                # pad() adds a 1 at the end for shape but [:, 0:2] removes it in the end
                good[:, 0, :] = np.dot(Kinv, pad(good[:, 0, :]).T).T[:, 0:2]
                good[:, 1, :] = np.dot(Kinv, pad(good[:, 1, :]).T).T[:, 0:2]
                
                model, inliers = ransac((good[:, 0], good[:, 1]),
                                        EssentialMatrixTransform,
                                        min_samples=8,
                                        residual_threshold=0.5,
                                        max_trials=200)
                good = good[inliers]
                #print(sum(inliers), " matches")


            for (kp1, kp2) in good:
                # denormalize the points
                c1 = np.dot(K, np.array([kp1[0], kp1[1], 1.0]))
                c2 = np.dot(K, np.array([kp2[0], kp2[1], 1.0]))

                x1, y1 = int(round(c1[0])), int(round(c1[1]))
                x2, y2 = int(round(c2[0])), int(round(c2[1]))
                cv2.circle(frame, (x1, y1),1,(0,255,0), 1)
                cv2.line(frame, (x1, y1), (x2, y2), (255,0,0), 1)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    #stop = time.time()
    #print("FULL FRAME: ", stop-start_full, "\n")

vid.release()
cv2.destroyAllWindows()

