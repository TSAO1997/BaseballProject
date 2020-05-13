from collections import deque
import numpy as np
import cv2
import imutils
import sys

video_path = './170929_cam_2.avi'
#video_path = './170929_cam_2.avi'
#initial_box = (494, 306, 60, 60)
video = cv2.VideoCapture(video_path)
tracker = cv2.TrackerMOSSE_create()
initBB = None
pts = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('170929_test.avi', fourcc, 30.0, (720, 540))
pre_center = (0,0)
count_miss = 0
while True:
    ret, frame = video.read()
    if frame is None:
        break
    frame = imutils.resize(frame, 720, 540)
    (H,W) = frame.shape[:2]
    if initBB:
        success, box = tracker.update(frame)
        overlay = frame.copy()
        alpha = 0.6
        if (success and count_miss <5):
            center = (int(box[0]+box[2]/2), int(box[1]+box[3]/2))
            if (pre_center == center):
                count_miss += 1
                continue
            pts.append(center)
            pre_center = center
            left_top_point = (int(box[0]), int(box[1]))
            right_bottom_point = (int(box[0]+box[2]), int(box[1]+box[3]))
            cv2.rectangle(frame, left_top_point, right_bottom_point, (0,255,0),2)
        else:
            count_miss += 1
        for i in range(1,len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (0,0,255), 5)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.imshow("Tracking", frame)
    out.write(frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        init_center = (int(initBB[0]+initBB[1]/2), int(initBB[2]+initBB[3]/2))
        pre_center = init_center
        tracker.init(frame, initBB)
    elif key == ord("q"):
        break
    #cv2.waitKey(10)
video.release()
out.release()
cv2.destroyAllWindows()