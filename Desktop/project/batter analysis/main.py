from example_modules import *
import sys

input_0 = '../../material/LHB_240FPS/143218.avi'
input_1 = '../../material/LHB_240FPS/143447.avi'

mono_clip_0 = read_clip_mono(input_0)
mono_clip_1 = read_clip_mono(input_1)

hist ,thres = 8, 128
#!! 如果有重複使用的code盡量函示化
ball_detector_0 = MovingBallDetector(mono_clip_0[0],hist=hist, thres=thres, kr=3)
temp_kp_0 = []
keyframe_0 = 0
for i ,frame in enumerate(mono_clip_0):
    fgmask_0 = ball_detector_0.gen_differential_img(frame, mog=True)
    blob,kps = ball_detector_0.draw_blob_detected_ball_on_img(fgmask_0)
    if not kps:
        continue
    elif not temp_kp_0:
        temp_kp_0.append((i,kps[0].pt[0]))
    else:
        #!! 在稍稍複雜要多花幾秒去閱讀的判斷式，可以先用命名代替註解，例如
        #!! ball_is_going_left_nearby = (temp_kp_0[0][1]>kps[0].pt[0] and abs(temp_kp_0[0][1]-kps[0].pt[0])<=40)
        #!! if ball_is_going_left_nearby: .... 
        if (temp_kp_0[0][1]>kps[0].pt[0] and abs(temp_kp_0[0][1]-kps[0].pt[0])<=40):
            keyframe_0 = int((i+temp_kp_0[0][0])/2)
            break
        else:
            #!! temp_kp_0[0] = (i,kps[0].pt[0])   看起來只有一項似乎不需用list?
            temp_kp_0.pop()
            temp_kp_0.append((i,kps[0].pt[0]))
print(keyframe_0)
#!! 同上
ball_detector_1 = MovingBallDetector(mono_clip_1[0],hist=hist, thres=thres, kr=3)
temp_kp_1 = []
keyframe_1 = 0
for i ,frame in enumerate(mono_clip_1):
    fgmask_1 = ball_detector_1.gen_differential_img(frame, mog=True)
    blob,kps = ball_detector_1.draw_blob_detected_ball_on_img(fgmask_1)
    if not kps:
        continue
    elif not temp_kp_1:
        temp_kp_1.append((i,kps[0].pt[0]))
    else:
        if (temp_kp_1[0][1]>kps[0].pt[0] and abs(temp_kp_1[0][1]-kps[0].pt[0])<=40):
            keyframe_1 = int((i+temp_kp_1[0][0])/2)
            break
        else:
            temp_kp_1.pop()
            temp_kp_1.append((i,kps[0].pt[0]))
print(keyframe_1)

end_frame = min(keyframe_0,keyframe_1)
#!! 因為原檔是灰階影像，用上面的mono_clip_0 就可以了，不用重新讀rgb
clip_0 = read_clip_rgb(input_0)[abs(end_frame-keyframe_0):keyframe_0+abs(end_frame-keyframe_0)]
clip_1 = read_clip_rgb(input_1)[abs(end_frame-keyframe_1):keyframe_1+abs(end_frame-keyframe_1)]

#!! 下面影片會結束在擊球點，就應用而言，會希望再看一下，觀察揮棒的延伸軌跡以及擊球結果
for i in range(end_frame):
    output = np.hstack((clip_0[i],clip_1[i]))
    cv2.imshow('Compare', output)
    cv2.waitKey(15)
    if i == end_frame-1:
        cv2.waitKey(0)
