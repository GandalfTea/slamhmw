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


if len(sys.argv) < 2:
    print("Error: Video argument not found.\n")
    sys.exit()


Wf = 1920 
Hf = 1080

#GW = 254
#GH = 240
GW = 1920 // 12
GH = 1080 // 12

class Frame(object):
    def __init__(self, img):
        self.img = img
    
        # Extract and compute
        self.kp = self.extract(self.img) 
        _, self.des = sift.compute(self.img, self.kp)

        # Camera Intrinsics
        # by model values, F should be 100
        self.F = 450
        self.K = np.array([[self.F, 0, self.img.shape[1]//2],
                           [0, self.F, self.img.shape[0]//2],
                           [0, 0, 1]])
        self.Kinv = np.linalg.inv(self.K)


    def extract(self, img):
        kps= []
        for rw in range(0, Wf, GW):
            for rh in range(0, Hf, GH):
                a = img[rh:rh+GH, rw:rw+GW]
                cv2.circle(frame, (rw, rh),1,(0,0,0), 2)
                pts = cv2.goodFeaturesToTrack(a, 3000, qualityLevel=0.1, minDistance=7)

                if pts is not None:
                    # recalibrating data for bigger frame 
                    kp = [cv2.KeyPoint(x=f[0][0] + rw, y=f[0][1] + rh, size=20) for f in pts]
                    kps.extend(kp)
        return kps


# turn [x, y] -> [x, y, 1]
def pad(p):
    return np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)


# Normalized coordinates are independent of the resolution of the image and give better 
# numerical stability for some multi-view geometry algorithms than pixel coordinates.
# Normalized points are stored in values -1 - 1, with 1 being the larger side of the picture
# The origin is the middle of the image. 
# first one is changing the queryIdx, second trainIdx

def normalize(coords, Kinv):
    # order of change is queryIdx followed by trainIdx
    coords[:, 0, :] = np.dot(Kinv, pad(coords[:, 0, :]).T).T[:, 0:2]
    coords[:, 1, :] = np.dot(Kinv, pad(coords[:, 1, :]).T).T[:, 0:2]
    return coords


# Extract R (rotation) and t (translation) from E (EssentialMatrix)
def compute_Rt(E):

    # property of W:  W^-1 = W.T
    W = np.mat([[0, -1, 0],[1, 0, 0], [0, 0, 1]], dtype=float)

    # Single value decomposition. Results:
    # * U, Vt  -- orthogonal 3x3
    # * e      -- diagnoal 3x3 with:
    #         [ s 0 0
    #           0 s 0
    #           0 0 0 ]
    # where s is singular value of E

    U, e, Vt = np.linalg.svd(E.params)

    
    assert np.linalg.det(U) > 0    # det of [[a,b][c,d]] is ad - bc
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    # R = U * W.T * e * U.T
    R = np.dot(np.dot(U, W.T), Vt)
    if np.sum(R.diagonal()) < 0:    # W.T sometimes gives wierd value, so choose between .T or not
        R = np.dot(np.dot(U, W), Vt)

    # t = U * W * e * V.T
    t = U[:, 2]

    # concat them into one single matrix [3, 4]
    Rt = np.concatenate([R, t.reshape(3,1)], axis=1)
    return Rt


def matchAndDraw(f1, f2, frame):
    matches = bf.knnMatch(np.array(f1.des), np.array(f2.des), k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            if (m.distance < 32):
                good.append(( f1.kp[m.queryIdx].pt,  f2.kp[m.trainIdx].pt ))

    assert(len(set(good)) == len(good))

    if len(good) > 8:
        good = np.array(good)
        good = normalize(good, f1.Kinv)

        model, inliers = ransac((good[:, 0], good[:, 1]),
                                EssentialMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.05,
                                max_trials=200)
        good = good[inliers]
        print(sum(inliers), " matches")

        Rt = compute_Rt(model)
        #print(Rt, end="\n\n")


    for (kp1, kp2) in good:
        # denormalize the points
        c1 = np.dot(f1.K, np.array([kp1[0], kp1[1], 1.0]))
        c2 = np.dot(f1.K, np.array([kp2[0], kp2[1], 1.0]))

        # Draw
        x1, y1 = int(round(c1[0])), int(round(c1[1]))
        x2, y2 = int(round(c2[0])), int(round(c2[1]))
        cv2.circle(frame, (x1, y1),1,(255,0,0), 1)
        cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 1)

    return Rt 


vid = cv2.VideoCapture(sys.argv[1])

# Extractors and Matches
orb = cv2.ORB_create()
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

if (vid.isOpened() == False):
    print("Error opening video.");
    sys.exit()


frames = [] 
while(vid.isOpened()):
    start_full = time.time()

    ret, frame = vid.read()
    #frame = cv2.resize(frame, (896, 504), interpolation = cv2.INTER_AREA)
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == True:
        frames.append(Frame(gray))

        if len(frames) > 1:
            # matches
            f1 = frames[-1]
            f2 = frames[-2]

            Kt = matchAndDraw(f1, f2, frame)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    #stop = time.time()
    #print("FULL FRAME: ", stop-start_full, "\n")

vid.release()
cv2.destroyAllWindows()

