import os
import time
import operator
import imutils

import cv2
import numpy as np

# CAMERA = [0]
CAMERA = [0, 1, 2, 3]

class main_video:
    def preprocess(raws):
        imgs = []
        for raw in raws:
            img = raw
            # img = cv2.resize(img, dsize=(256, 144), interpolation=cv2.INTER_CUBIC)    # 16:9
            # img = cv2.resize(img, dsize=(512, 288), interpolation=cv2.INTER_CUBIC)    # 16:9
            # img = cv2.resize(img, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)    # 4:3
            img = cv2.resize(img, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)      # 4:3
            # img = imutils.rotate_bound(img, 90)

            imgs.append(img)
            
        if len(imgs) == 1:
            image = imgs[0]
        if len(imgs) >= 2:
            image = np.hstack((imgs[0], imgs[1]))
        if len(imgs) == 4:
            image2 = np.hstack((imgs[2], imgs[3]))
            image = np.vstack((image, image2))
            
        return image
        
    def __init__(self, camera=1):
        fps_time = 0
        frame = 0
        avg_fps = 0
        his_fps = []
        
        cams = [cv2.VideoCapture(cam) for cam in CAMERA]
        
        imgs = []
        for i, cam in enumerate(cams):
            ret_val, img = cam.read()
            imgs.append(img)
            
        image = main_video.preprocess(imgs)
        
        # h, w, c = image_raw.shape
        # h2, w2, c2 = image2_raw.shape
        
        # print(h, w, c, h2, w2, c2)
        
        # Main loop
        while True:
            imgs = []
            
            for i, cam in enumerate(cams):
                ret_val, img = cam.read()
                imgs.append(img)
                
            image = main_video.preprocess(imgs)
            
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            
            print(ret_val, "%.2f" % fps)
            his_fps.append(fps)
            
            frame += 1
            if frame > 120:
                avg_fps = sum(his_fps) / len(his_fps)
                frame = 0
                his_fps = []
            
            # self.display_all(image, fps)
            self.display_all(image, avg_fps)
            
            if cv2.waitKey(1) == 27:
                break
            
        cv2.destroyAllWindows()
        
    def display_all(self, image, fps):
        
        h, w, c = image.shape
        
        cv2.putText(image,
            "FPS: %f" % fps,
            (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)
        cv2.putText(image,
            "RES: %dx%d" % (w, h),
            (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)
        
        cv2.imshow('Bedssys', image)
        
if __name__ == '__main__':
    main_video()

