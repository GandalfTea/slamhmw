#!usr/bim/python3

import cv2
import numpy as np
import sys
from skimage.measure import ransac 
from skimage.transform import FundamentalMatrixTransform

# video split
W = 1920
H = 1080

# 254, 240
GW = 1920
GH = 1080

if len(sys.argv) < 2:
    print("Error: Video argument not found.\n")
    sys.exit()

orb = cv2.ORB_create()
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()
vid = cv2.VideoCapture(0)

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
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(frames) > 2: frames.pop(0)

    if ret == True:
        kps= []
        for rw in range(0, W, GW):
            for rh in range(0, H, GH):
                a = gray[rh:rh+GH, rw:rw+GW]
                cv2.circle(frame,(rw, rh),1,(0,0,0), 2)
                pts = cv2.goodFeaturesToTrack(a, 3000, qualityLevel=0.05, minDistance=7)

                if pts is not None:
                    # recalibrating data 
                    kp = [cv2.KeyPoint(x=f[0][0] + rw, y=f[0][1] + rh, size=20) for f in pts]
                for p in kp:
                    kps.append(p)
        kp, des = sift.compute(gray, kps)
        frames.append(Frame(gray, kps, des))

        if len(frames) > 1:
            # matches
            f2 = frames[-2]
            des_2 = np.array(f2.des)

            matches = bf.knnMatch(des, des_2, k=2)

            good = []
            idx1 , idx2 = [], []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    if (m.distance < 32):
                        p1 = kps[m.queryIdx].pt
                        p2 = f2.kp[m.trainIdx].pt
                        good.append((p1, p2))

            assert(len(set(good)) == len(good))

            if len(good) > 8:
                good = np.array(good)
                print(good.shape)
                
                model, inliers = ransac((good[:, 0], good[:, 1]),
                                        FundamentalMatrixTransform,
                                        min_samples=8,
                                        residual_threshold=1,
                                        max_trials=100)
                good = good[inliers]
                print(sum(inliers))

            for (kp1, kp2) in good:
                kp1s = (int(round(kp1[0])), int(round(kp1[1])))
                kp2s = (int(round(kp2[0])), int(round(kp2[1])))
                cv2.line(frame, kp1s, kp2s, (255,0,0), 2)

        for p in kps:
            x, y = int(p.pt[0]), int(p.pt[1]) 
            cv2.circle(frame, (x, y),1,(0,255,0), 1)



        """

        f1 = frames[-1]
        f2 = frames[-2]

        pts_1 = cv2.goodFeaturesToTrack(f1, 3000, qualityLevel=0.01, minDistance=7)
        pts_2 = cv2.goodFeaturesToTrack(f2, 3000, qualityLevel=0.01, minDistance=7)
        #kp_1 = orb.detect(f1, None);
        #kp_2 = orb.detect(f2, None);

        #if pts_1 is not None or pts_2 is not None:
        if True:
            kp_1 = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts_1]
            kp_2 = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts_2]
            kp, des_1 = orb.compute(f1, kp_1)
            kp, des_2 = orb.compute(f2, kp_2)
        """

        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break


vid.release()
cv2.destroyAllWindows()

