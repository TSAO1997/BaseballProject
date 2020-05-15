from collections import deque
import numpy as np
import cv2
import imutils
import sys

video_path = './170929_cam_2.avi'
#video_path = './170929_cam_2.avi'

video = cv2.VideoCapture(video_path)
tracker = cv2.TrackerMOSSE_create()
#initial Boundary Box
initBB = None
#紀錄每個frame球的座標
pts = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('170929_test.avi', fourcc, 30.0, (720, 540))
#下面兩個主要是用來記一些東西來做判斷讓呈現效果好一點
#pre_center是用來記錄一個frame的center座標，由於有時候進到球進到手套之後會發生tracker還在運作，因此透過這個來判斷球是否在原地太多frame決定要不要停止追蹤
#count_miss原本是拿來紀錄tracker失敗的次數，這也是用來做尾端的處理，有情況是球投完tracker fail幾個frame之後又突然開始，因此這拿來讓miss幾個之後就能直接斷掉
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
            #因為boundary box的return值是(左上點的x,左上點的y, w, h)，所以下面的兩點才會長那樣
            left_top_point = (int(box[0]), int(box[1]))
            right_bottom_point = (int(box[0]+box[2]), int(box[1]+box[3]))
            cv2.rectangle(frame, left_top_point, right_bottom_point, (0,255,0),2)
        else:
            count_miss += 1
        #把軌跡畫出來
        for i in range(1,len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (0,0,255), 5)
        #這一行是對frame做alpha transparency，讓那個軌跡線可以變比較淡
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.imshow("Tracking", frame)
    out.write(frame)
    #按s可以暫停並選取roi
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
        init_center = (int(initBB[0]+initBB[1]/2), int(initBB[2]+initBB[3]/2))
        pre_center = init_center
        tracker.init(frame, initBB)
    elif key == ord("q"):
        break

video.release()
out.release()
cv2.destroyAllWindows()