import numpy  as np
import math
import cv2
import time

def read_clip_mono(path):
    cap = cv2.VideoCapture(path)
    clip_buf=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break 
        mono = frame[:,:,1]
        clip_buf.append(mono)
    return clip_buf

def read_clip_rgb(path):
    cap = cv2.VideoCapture(path)
    clip_buf=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break 
        clip_buf.append(frame)
    return clip_buf

class MovingBallDetector(object):
    def __init__(self, frame, hist=8, thres=16, kr=7):
        self.WINDOW_NAME = "Example image"
        self.roi = self.cut_roi(frame)
        self.H, self.W = self.roi.shape 
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=hist, varThreshold=thres, detectShadows=False) 
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kr,kr))  
        blob_params = self.set_blob_params()
        self.blob_detector = cv2.SimpleBlobDetector_create(blob_params)

    def cut_roi(self, img):
        return img[0:540,0:720]

    def gen_differential_img(self, frame, mog=False):
        fgmask = self.fgbg.apply(frame) 
        if mog:
            fgmask_mog2 = fgmask
            fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_CLOSE, self.kernel) 
            fgmask_mog2 = cv2.morphologyEx(fgmask_mog2, cv2.MORPH_OPEN, self.kernel) 
            return fgmask_mog2
        return fgmask
    
    def set_blob_params(self):
        ball_r = 10 
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 100
        # Filter by Area.
        params.filterByArea = True  # radius = 20~30 pixels (plate width = 265 pixels)
        params.minArea = ball_r*ball_r*math.pi *0.5 #    
        params.maxArea = ball_r*ball_r*math.pi *1.8 # 
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.6
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.6
        return params

    def draw_blob_detected_ball_on_img(self, img):
        inv_img = cv2.bitwise_not(img)
        keypoints = self.blob_detector.detect(inv_img)
        img = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for kp in keypoints:
            #print("kp.size="+str(kp.size))
            print("kp.xy = (%d, %d)"%(kp.pt[0], kp.pt[1] ))
        return img ,keypoints

    def demo_video(self, clip):
        for i, frame in enumerate(clip):
            fgmask = self.gen_differential_img(frame, mog=True)
            blob,kps = self.draw_blob_detected_ball_on_img(fgmask)
            cv2.imshow(self.WINDOW_NAME, blob)
            cv2.waitKey(1)

def run_param_for_bgs():
    path = ('material/LHB_240FPS/Lin_toss_1227 (2).avi')
    clip_buf = read_clip_mono(path)
    frame_total= len(clip_buf)
    for thres in [128]:
        for hist in [64]:
            t0=time.time()
            print("hist = "+str(hist), "thres = "+str(thres))
            ball_detector = MovingBallDetector(clip_buf[0],hist=hist, thres=thres, kr=3)
            ball_detector.demo_video(clip_buf[0:frame_total])
            print("ms per frame: "+str((time.time()-t0)*1000/frame_total))    
    pass

#run_param_for_bgs()
# test_clip()