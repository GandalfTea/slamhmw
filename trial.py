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

IRt = np.eye(4)
# Extractors and Matches
#orb = cv2.ORB_create()
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

POINTS_TO_RENDER = [6, 9]

class Frame(object):
    def __init__(self, img):
        self.img = img
        # Camera Intrinsics
        # by model values, F should be 100
        self.F = 450
        self.K = np.array([[self.F, 0, self.img.shape[1]//2],
                           [0, self.F, self.img.shape[0]//2],
                           [0, 0, 1]])
        self.Kinv = np.linalg.inv(self.K)
        self.pose = IRt

        # Extract and compute
        kp, pts  = self.extract(self.img) 
        self.kp = kp
        self.pts = normalize(pts, self.Kinv)
        #start = time.time() # debug
        if self.kp.any() : _, self.des = sift.compute(self.img, self.kp)
        #stop = time.time()
        #print("SIFT Compute : ", stop - start)



    def extract(self, img):
        kps= []
        for rw in range(0, Wf, GW):
            for rh in range(0, Hf, GH):
                a = img[rh:rh+GH, rw:rw+GW]
                #cv2.circle(frame, (rw, rh),1,(0,0,0), 2)
                pts = cv2.goodFeaturesToTrack(a, 3000, qualityLevel=0.1, minDistance=7)

                if pts is not None:
                    # recalibrating data for bigger frame 
                    kp = [cv2.KeyPoint(x=f[0][0] + rw, y=f[0][1] + rh, size=20) for f in pts]
                    kps.extend(kp)
        if kps is not None:
            pts = []
            for kp in kps:
                pts.append(([kp.pt[0], kp.pt[1]]))
            return np.array(kps), np.array(pts)
        else:
            return 0


class Point(object):
    def __init__(self, loc):
        self.location = loc
        self.frames = []
        self.idxs = []

    def observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


# Triangulation to get 3-D points
#    pose1, pose2  (4 x 4)
#    kp1, kp2      (N x 2): normalized keypoints
#  Returns:
#    X  (3 x N): 3-D world coords
#    X1 (3 x N): 3-D view1 coords
#    X2 (3 x N): 3-D view2 coords
def triangulate(pose1, pose2, kp1, kp2):
    #kp1_3D = np.ones((3, kp1.shape[0]))
    #kp2_3D = np.ones((3, kp2.shape[0]))
    #kp1_3D, kp1_3D = kp1[:, 0].copy(), kp1[:, 1].copy()
    #kp2_3D, kp2_3D = kp2[:, 0].copy(), kp2[:, 1].copy()
    
    #X = cv2.triangulatePoints(pose1[:3], pose2[:3], kp1.T, kp2.T).T
    #X /= X[3]

    #X1 = np.dot(pose1[:3], X.T)
    #X2 = np.dot(pose2[:3], X.T)

    #return X[:3], X1, X2

    return cv2.triangulatePoints(pose1[:3], pose2[:3], kp1.T, kp2.T).T



# turn [x, y] -> [x, y, 1]
def pad(p):
    return np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)


# Normalized coordinates are independent of the resolution of the image and give better 
# numerical stability for some multi-view geometry algorithms than pixel coordinates.
# Normalized points are stored in values -1 - 1, with 1 being the larger side of the picture
# The origin is the middle of the image. 

def normalize(coords, Kinv):
    # order of change is queryIdx followed by trainIdx
    #coords[:, 0, :] = np.dot(Kinv, pad(coords[:, 0, :]).T).T[:, 0:2]
    #coords[:, 1, :] = np.dot(Kinv, pad(coords[:, 1, :]).T).T[:, 0:2]
    #return coords
    return np.dot(Kinv, pad(coords).T).T[:, 0:2]

def denormalize(coords, K):
    ret = np.dot(K, np.array([coords[0], coords[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


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
    #  ? why skip e
    R = np.dot(np.dot(U, W.T), Vt)
    if np.sum(R.diagonal()) < 0:    # W.T sometimes gives wierd value, so choose between .T or not
        R = np.dot(np.dot(U, W), Vt)

    # t = U * W * e * V.T
    t = U[:, 2]

    # concat them into one single matrix [3, 4]
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    return Rt


def matchAndDraw(f1, f2, frame):
    matches = bf.knnMatch(np.array(f1.des), np.array(f2.des), k=2)

    good = []
    idx1, idx2 = [], []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            if (m.distance < 32):
                good.append(( f1.pts[m.queryIdx],  f2.pts[m.trainIdx] ))
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

    #assert(len(set(good)) == len(good))

    if len(good) > 8:
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        good = np.array(good)

        model, inliers = ransac((good[:, 0], good[:, 1]),
                                EssentialMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.005,
                                max_trials=200)
        good = good[inliers]

        Rt = compute_Rt(model)
        #print(Rt, end="\n\n")

    #for (kp1, kp2) in good:
    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        x1, y1 = denormalize(pt1, f1.K)
        x2, y2 = denormalize(pt2, f2.K)

        # denormalize the points
        #c1 = np.dot(f1.K, np.array([kp1[0], kp1[1], 1.0]))
        #c2 = np.dot(f1.K, np.array([kp2[0], kp2[1], 1.0]))

        # Draw
        #x1, y1 = int(round(c1[0])), int(round(c1[1]))
        #x2, y2 = int(round(c2[0])), int(round(c2[1]))
        cv2.circle(frame, (x1, y1),1,(255,0,0), 2)
        cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 2)

    return idx1[inliers], idx2[inliers], Rt 

class Slam():
    def __init__(self):
        self.points = []
        self.frames = [] 

    def run(self, video, queue):
        global POINTS_TO_RENDER
        vid = cv2.VideoCapture(video)


        if (vid.isOpened() == False):
            print("Error opening video.");
            sys.exit()

        while(vid.isOpened()):
            start_full = time.time()

            ret, frame = vid.read()
            #frame = cv2.resize(frame, (896, 504), interpolation = cv2.INTER_AREA)
            frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2), interpolation = cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if ret == True:
                self.frames.append(Frame(gray))

                if len(self.frames) > 1:
                    # matches
                    f1 = self.frames[-1]
                    f2 = self.frames[-2]

                    if not f1.kp.any() or not f2.kp.any():
                        sys.exit()

                    idx1, idx2, Rt = matchAndDraw(f1, f2, frame)
                    f1.pose = np.dot(Rt, f2.pose)

                    # 3D coords
                    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
                    pts4d /= pts4d[:, 3:]
                    pts4d[1][2] *= 100

                    # potential good point mask
                    #mask = cv2.where(pts_3D[3] != 0)[0]

                    # filter the points behind the camera
                    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

                    for i,p in enumerate(pts4d):
                        if not good_pts4d[i]:
                            continue
                        if p[2] > 0:
                            pt = Point(p)
                            pt.observation(f1, idx1[i])
                            pt.observation(f2, idx2[i])
                            self.points.append(pt)

                poses, pts = [], []
                for f in self.frames:
                    poses.append(f.pose)
                for p in self.points:
                    #print(p.location)
                    pts.append(p.location)
                state = poses, pts

                queue.put(state)


                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break

        vid.release()
        cv2.destroyAllWindows()

