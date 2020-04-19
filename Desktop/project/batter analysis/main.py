from example_modules import *
import sys

input_0 = '../material/LHB_240FPS/143218.avi'
input_1 = '../material/LHB_240FPS/143447.avi'

mono_clip_0 = read_clip_mono(input_0)
mono_clip_1 = read_clip_mono(input_1)
if not mono_clip_0 or not mono_clip_1:
    print("Fail to read input")

hist ,thres = 8, 128

def get_keyframe(mono_clip):
    ball_detector = MovingBallDetector(mono_clip[0],hist=hist, thres=thres, kr=3)
    temp_kp_x = (-1,-1)
    keyframe = 0
    for i ,frame in enumerate(mono_clip):
        fgmask = ball_detector.gen_differential_img(frame, mog=True)
        blob,kps = ball_detector.draw_blob_detected_ball_on_img(fgmask)
        if not kps:
            continue
        else:
            #!! 在稍稍複雜要多花幾秒去閱讀的判斷式，可以先用命名代替註解，例如
            ball_is_going_left_nearby = (temp_kp_x[1]>kps[0].pt[0] and abs(temp_kp_x[1]-kps[0].pt[0])<=40)
            if ball_is_going_left_nearby:
                keyframe = int((i+temp_kp_x[0])/2)
                return keyframe
            else:
                #!! temp_kp_0[0] = (i,kps[0].pt[0])   看起來只有一項似乎不需用list?
                temp_kp_x = (i,kps[0].pt[0])
    

keyframe_0 = get_keyframe(mono_clip_0)
keyframe_1 = get_keyframe(mono_clip_1)
end_frame = min(keyframe_0,keyframe_1)

clip_0 = mono_clip_0[abs(end_frame-keyframe_0):keyframe_0+abs(end_frame-keyframe_0)]
clip_1 = mono_clip_1[abs(end_frame-keyframe_1):keyframe_1+abs(end_frame-keyframe_1)]

#!! 下面影片會結束在擊球點，就應用而言，會希望再看一下，觀察揮棒的延伸軌跡以及擊球結果
for i in range(end_frame):
    output = np.hstack((clip_0[i],clip_1[i]))
    cv2.imshow('Compare', output)
    cv2.waitKey(15)
    if i == end_frame-1:
        cv2.waitKey(0)
