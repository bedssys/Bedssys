import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
from random import randint
import argparse
import logging
import time
import operator
import imutils

import cv2
import numpy as np
import math

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from itertools import chain, count
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import winsound

import darknet.json as dk
import facerec.recognize as fr
# import deepface.deepface as df

import security

## Input management
CAMERA = []     # Default value, if no camera is given, switch to video mode

VIDEO = "utilities/test_vid.mp4"
REAL_FPS = 6
PROC_FPS = 3    # Proc is surely < Real
SKIP_FRAME = round(REAL_FPS/PROC_FPS) - 1

# 5th is face camera. Remove to use cailing cams cropped by FREG.
# CAMERA = [0]
# CAMERA = [0, 1]
# CAMERA = [cv2.CAP_DSHOW + 0]    # Using directshow to fix black bar
# CAMERA = ["rtsp://167.205.66.187:554/onvif1"]
# CAMERA = [  "rtsp://167.205.66.147:554/onvif1",
            # "rtsp://167.205.66.148:554/onvif1",
            # "rtsp://167.205.66.149:554/onvif1",
            # "rtsp://167.205.66.150:554/onvif1",
            # cv2.CAP_DSHOW + 0                   ]
# CAMERA = [  "rtsp://192.168.0.108:554/onvif1", 88:83:5D:AA:B5:52
            # "rtsp://192.168.0.107:554/onvif1", 00:7E:56:B0:78:8A
            # "rtsp://192.168.0.104:554/onvif1", 00:7E:56:B0:63:8A	
            # "rtsp://192.168.0.110:554/onvif1", 00:7E:56:B0:78:18
            # cv2.CAP_DSHOW + 0                   ]
# CAMERA = [  "rtsp://192.168.0.108:554/onvif1",
            # "rtsp://192.168.0.107:554/onvif1",
            # "rtsp://192.168.0.104:554/onvif1",
            # "rtsp://192.168.0.110:554/onvif1",
            # cv2.CAP_DSHOW + 0                   ]
# CAMERA = [  "rtsp://192.168.0.100:554/onvif1",
            # "rtsp://192.168.0.101:554/onvif1",
            # "rtsp://192.168.0.103:554/onvif1",
            # "rtsp://192.168.0.102:554/onvif1",
            # cv2.CAP_DSHOW + 0                   ]

FPSLIM = 0  # Set to 0 for unlimited
            
# Size of the images, act as a boundary
IMAGE = [1024,576]
SUBIM = [512,288]

ROTATE = [0, 0, 0, 0, 270]
# ROTATE = [180, 180, 180, 180, 90]

# Face camera, the fifth camera on the list
FCAMDS = 1                          # Face camera downscale
# FCAMCP = [0.2, 1-0.5, 0.2, 1-0.2]   # Crop fraction from top, bottom, left, right
FCAMCP = [0.35, 1-0.25, 0.2, 1-0.2]   # Crop fraction from top, bottom, left, right
FCOFF = SUBIM                       # Center location of face camera



## System-wide parameters
# Disable/Enable the actual systems and not just visual change
SYS_OPOSE = True
SYS_ACT = SYS_OPOSE and True
# SYS_ACT = True
SYS_DARK = False
SYS_FACEREC = True

# OpenPose model parameters
# OPSIZE = "256x144"
# OPSIZE = "512x288"
# OPSIZE = "717x403"
# OPSIZE = "768x432"
OPSIZE = "1024x576"
# OPSIZE = "1280x720"
# OPSIZE = "1536x864"
# OPSIZE = "2048x1152"
OPGPU = 0./6.0 # GPU fraction limit
# LSGPU = 0./6.0
# OPGPU = 1/6.0 # GPU fraction limit
LSGPU = 0/6.0



## LSTM Parameters
# N_STEPS = 8
N_STEPS = 5
# DATASET_PATH = "data/"
# DATASET_PATH = "data/Overlap_fixed/"
# DATASET_PATH = "data/Overlap_fixed4/"
# DATASET_PATH = "data/Overlap_fixed4_separated/"
# DATASET_PATH = "data/2a_Amplify/"
# DATASET_PATH = "data/Direct2a/"
# DATASET_PATH = "data/Direct2a/Normalize/"
# DATASET_PATH = "data/Direct2a/NormalizePoint/"
DATASET_PATH = "data/Direct2a/NormalizeOnce/"
# DATASET_PATH = "data/Test/5/"

LAYER = 2   # 1: Default [36,36] # 2: Simpler [36]

## Preprocessing schemes, only applies right before the poses loaded to LSTM.
# No effect to the original pose data.
# Group A, main preprocessing:
# 1: Amplify        - Poses emulated as if there's a big border between sub-images
# 2: Normalize      - Individual pose returned to origin
# 3: NormalizeOnce  - Every pose in a gesture will be relative to the first in the gesture
# 4: NormalizePoint - Every point in a gesture will be relative to the first point in the gesture
# 5: Reverse        - Poses in 4 sub-images emulated as if happening in a single image
# Other: No preprocessing
POSEAMP = 1000  # [Amplify] Value added if a pose is over the sub-image boundary

# Group B, idle management:
# 1: Null    - Unmoving gestures (average) are forced to be all null
# 2: Null    - Unmoving gestures (key point [neck, or nose]) are forced to be all null
# Other: No preprocessing
IDLETH = int(IMAGE[0]/100)  # Max distance (in coord) a gesture forced to be idling

PREPROC = [3,1]

## Label id selection schemes
# No effect to the original pose data. Based on the index:
# 0: Weighted   - Positive poses receive boosted confidence (lowering false "suspicious").
# 1: Grouped    - Big gesture (DR, UR, DL, UL, ND) will be groups, averaged, max obtained.
#                 Labels in losing groups will be totally ignored (zero)
# After: Max confidence
LABSEL = [True,False]

# Label weight for weighted label scheme, multiplied to the base confidence
LABWEI = np.array([1,1,1,1,  0,0,0,0,  0,0,0,0,  0,0,0,0,  0]) * 0.2 + 1
# LABWEI = np.array([1,1,1,1,  0,0,0,0,  0,0,0,0,  0,0,0,0]) * 0.2 + 1
LABGRO = [  [0,4,8,12], 
            [1,5,9,13],  
            [2,6,10,14],  
            [3,7,11,15],  
            [16]]

LABELS = [
    "jalan_DR", "jalan_UR", "jalan_DL", "jalan_UL",
    "barang2_DR", "barang2_UR", "barang2_DL", "barang2_UL",
    "barang1l_DR", "barang1l_UR", "barang1l_DL", "barang1l_UL",
    "barang1r_DR", "barang1r_UR", "barang1r_DL", "barang1r_UL",
    "diam_ND"
]

# LABELS = [    
    # "jalan_NE", "jalan_NW", "jalan_SE", "jalan_SW",
    # "menyapu_NE", "menyapu_NW", "menyapu_SE", "menyapu_SW",
    # "barang_NE", "barang_NW", "barang_SE", "barang_SW",
    # "diam_NE", "diam_NW", "diam_SE", "diam_SW"
# ]

# LABELS = ["normal", "anomaly"]



## Security Parameters
N_HIST = 10
FRPARAM = 0.3   # Individual frame parameter, depending on the post processing used.
HISTH = 0.5     # Historical threshold for final trigger.

## Postprocessing schemes, historical level calculation
# Before: N_HIST frames collected, each having percentage of positive detections vs. all detections
# 0: Count threshold    - Percentage of frames above PARAM threshold vs. all frames.
# 1: Average            - Average all frames (no PARAM required)
# 2: Percentile         - Calculate the PARAM percentile from all frames
# After: Check against historical threshold
POSTPROC = 2

# Alarms & indicators
ALDUR = 2                       # Alarm duration in seconds (using the file duration if it's shorter)
ALAUTH = 4                      # Authorized state duration, if there's any known face
ALSND = "utilities/alarm.wav"   # Alarm sound directory



## Utilities
# Prevent face blinking, hold prev result if new result is empty
HFACE = 0

# Prescale & Pratical face_reg region
FPSCALE = 1     # The face image prescale divisor
FUP = 2         # Facerec model upsample

# Cropping ceiling cams for face recog region
# FREG = [0, 200, 250, 800]                   # Face region, for single SW camera [y1, y2, x1, x2], 1024x576 single image
# FREG = [288+0, 288+100, 512+125, 512+340]   # Face region, for SW camera in 2x2 [y1, y2, x1, x2], 1024x576 four images
# FREG = [0, 576, 0, 1024]
# FREG = [350, 510, 400, 600]
FREG = [210, 360, 425, 590]

# Exit zone [y1, y2, x1, x2]
EX = [288,375,701,800]
EXR = 3     # Radius (square) from pose point to be used as color reference
EXTH = 0.2  # Threshold in distance fraction

# Masking areas to NOT be detected by openpose.
# Used to hide noisy area unpassable by human. (Masks are not shown during preview)
# The mask is a polygon, specify the vertices location.
DOMASK = 1
DRAWMASK = 0    # Preview the masking or keep it hidden
# PMASK = [   np.array([[610,520],[770,430],[960,576],[660,576]], np.int32),       # SW
            # np.array([[185,430],[255,470],[70,570],[0,575],[0,530]], np.int32),  # SE
            # np.array([[760,200],[880,288],[1024,134],[985,44]], np.int32),       # NW
            # np.array([[260,190],[50,50],[136,53],[327,157]], np.int32)           # NE
            # ]   

# PMASK = [   np.array([[290,200],[0,0],[512,0],[350,180]], np.int32),               # NE
            # np.array([[650,200],[800,288],[1024,288],[1024,0],[985,44]], np.int32),         # NW
            # np.array([[185,430],[255,470],[70,570],[0,575],[0,300]], np.int32),    # SE
            # np.array([[610,520],[700,420],[770,380],[960,576],[660,576]], np.int32),          # SW
            # np.array([[950,400],[1024,400],[1024,500]], np.int32)]          # SW            
# PMASK = [   np.array([[290,200],[0,0],[512,0],[350,180]], np.int32),               # NE
            # np.array([[650,200],[800,288],[1024,288],[1024,0],[985,44]], np.int32),         # NW
            # np.array([[275,400],[190,400],[200,480],[270,460]], np.int32),    # SE
            # np.array([[185,430],[255,470],[70,570],[0,575],[0,300]], np.int32),    # SE
            # np.array([[900,576],[700,420],[640,400],[512,576]], np.int32),          # SW
            # np.array([[950,400],[1024,400],[1024,500]], np.int32)]          # SW
# PMASK = [ np.array([[0,0],[1024,0],[1024,576],[0,576]], np.int32) ]
PMASK = [ np.array([[0,0],[1,0],[1,1]], np.int32) ]
# PMASK = [   np.array([[579,500],[580,575],[760,574],[756,473],[724,443]], np.int32),
    # np.array([[384,339],[329,401],[154,343],[225,288],[386,287]], np.int32),
    # np.array([[960,478],[905,573],[1023,574],[1024,466]], np.int32),
    # np.array([[360,285],[393,229],[509,190],[511,365],[475,315]], np.int32),
    # np.array([[635,338],[706,374],[514,449],[516,364]], np.int32)]

DUMMY = False

SKX = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
SKY = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
            
class mainhuman_activity:

    # Pre-processing for every image
    def preprocess(raws, rots):
        imgs = []
        for img, rot in zip(raws, rots):
            img = cv2.resize(img, dsize=(SUBIM[0], SUBIM[1]), interpolation=cv2.INTER_CUBIC)  # 16:9
            
            # img = cv2.resize(img, dsize=(1024, 576), interpolation=cv2.INTER_CUBIC) # 16:9
            # img = cv2.resize(img, dsize=(512, 288), interpolation=cv2.INTER_CUBIC)  # 16:9
            # img = cv2.resize(img, dsize=(256, 144), interpolation=cv2.INTER_CUBIC)  # 16:9
            
            # img = cv2.resize(img, dsize=(464, 288), interpolation=cv2.INTER_CUBIC)  # 16:10
            # img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)  # 4:3
            # img = cv2.resize(img, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)  # 4:3
            # img = cv2.resize(img, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)  # 4:3
            img = imutils.rotate_bound(img, rot)

            imgs.append(img)
            
        if len(imgs) == 1:
            image = imgs[0]
        if len(imgs) >= 2:  # Two images side-by-side
            image = np.hstack((imgs[0], imgs[1]))
        if len(imgs) >= 4:  # Four images boxed
            image2 = np.hstack((imgs[2], imgs[3]))
            image = np.vstack((image, image2))
            
        return imgs, image
    
    def __init__(self, camera=CAMERA):
        
        self.fps = 1
        frame_time = 0
        hisfps = []     # Historical FPS data
        
        self.alprev = 0 # Prev alarm time
        self.altrig = 0 # Alarm triggered, -1 authorized, 0 neutral, 1 triggered
        
        freg = []
        
        if len(camera) > 0:
            from webcamvideostream import WebcamVideoStream
            cams = [WebcamVideoStream(src=cam, resolution=(1280,720)).start() for cam in camera]
            
            imgs = []
            for i, cam in enumerate(cams):
                # cam.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Internal buffer will now store only x frames
                img = cam.read()
                
                # If no image is acquired
                if (img is None):
                    # Black image
                    imgs.append(np.zeros((100,100,3), np.uint8))
                elif (img.size == 0):
                    imgs.append(np.zeros((100,100,3), np.uint8))
                else:
                    imgs.append(img)
                
            # TEST, 4 camera simulation
            # for i in range(3):
                # imgs.append(img)
                
            imgs, image = mainhuman_activity.preprocess(imgs, ROTATE)
            
            # Face camera, not rendered on main image
            if len(imgs) == 5:
                im_h, im_w = imgs[4].shape[:2]
                imf = imgs[4][round(im_h*FCAMCP[0]): round(im_h*FCAMCP[1]), round(im_w*FCAMCP[2]): round(im_w*FCAMCP[3])]   # Crop
                im_h, im_w = imf.shape[:2]
                imf = cv2.resize(imf, dsize=(round(im_w/FCAMDS), round(im_h/FCAMDS)), interpolation=cv2.INTER_CUBIC)  # Downsample
                im_h, im_w = imf.shape[:2]
                ky = 0 if im_h % 2 == 0 else 1
                kx = 0 if im_w % 2 == 0 else 1
                
                freg = [round(FCOFF[1]-im_h/2), round(FCOFF[1]+im_h/2)+ky, round(FCOFF[0]-im_w/2), round(FCOFF[0]+im_w/2)+kx]
            else:
                freg = FREG # Use cropped ceiling cams for face
        else:
            cams = []
            print("No camera given, trying to use video instead.")
            cap = cv2.VideoCapture(VIDEO, cv2.CAP_FFMPEG)
            time.sleep(1)
            if cap.isOpened() is False:
                print("Error opening video stream or file")
                return None
            frame = 0
            frame_skipped = 0
            ret_val, image = cap.read()
            freg = FREG # Use ceiling cams for face
        
        self.im_h, self.im_w = image.shape[:2]
        
        # print(h, w, c, h2, w2, c2)
                
        ###print("\n######################## Facerec")
        if SYS_FACEREC:
            facer = fr.face_recog(face_dir="./facerec/face/")
        
        ###print("\n######################## Darknet")
        if SYS_DARK:
            dark = dk.darknet_recog()
            ###print(dark.performDetect(image))
        
        ###print("\n######################## LSTM")
        if SYS_ACT:
            act = activity_human()
            act.test()
        
        ###print("\n######################## Openpose")
        if SYS_OPOSE:
            opose = openpose_human(image)
        
        # print("\n######################## Deepface")
        # dface = df.face_recog()
        # print(dface.run(image))
        
        hold_face = 0
        act_labs = []
        act_confs = []
        act_locs = []
        sec_hist = []
        sec_auths = {}
        
        if DUMMY:
            # Dummy pose
            dimg = cv2.imread("images/TestPose.jpg")
            doff_x = 0 
            doff_y = 30
            
            rimg = cv2.imread("images/Background.png")
        
        # For FPS calculation
        ptime = time.time()
        
        # Main loop
        while True:
            imgs = []
            if len(camera) > 0:
                for i, cam in enumerate(cams):
                    # Grab the frames AND do the heavy preprocessing for each camera
                    # ret_val, img = cam.read()
                    
                    # For better synchronization on multi-cam setup:
                    # Grab the frames first without doing the heavy stuffs (decode, demosaic, etc)
                    # ret_val = cam.grab()
                    
                    # The FIFO nature of the buffer means we can't get the latest frame
                    # Thus skip the earlier frames. Delay stats: 7s 8fps +artifact >>> 2s 3fps
                    # for i in range(5):
                        # ret_val = cam.grab()
                        
                    # Multi-threading using WebcamVideoStream
                    img = cam.read()
                    ###print(cam.grabbed)
                                
                    # If no image is acquired
                    if (img is None):
                        # Black image
                        imgs.append(np.zeros((100,100,3), np.uint8))
                    elif (img.size == 0):
                        imgs.append(np.zeros((100,100,3), np.uint8))
                    else:
                        imgs.append(img)
                    
                
                # for i, cam in enumerate(cams):
                    # # Decode the captured frames
                    # ret_val, img = cam.retrieve()
                    # imgs.append(img)
                
                # Skip frame if there's nothing
                if (imgs is [None]):
                    continue
                    
                # # TEST, 4 camera simulation
                # for i in range(3):
                    # imgs.append(img)
                    
                imgs, image = mainhuman_activity.preprocess(imgs, ROTATE)
                
                # Face camera, not seen on main image
                if len(imgs) == 5:
                    im_h, im_w = imgs[4].shape[:2]
                    imf = imgs[4][round(im_h*FCAMCP[0]): round(im_h*FCAMCP[1]), round(im_w*FCAMCP[2]): round(im_w*FCAMCP[3])]   # Crop
                    im_h, im_w = imf.shape[:2]
                    imf = cv2.resize(imf, dsize=(round(im_w/FCAMDS), round(im_h/FCAMDS)), interpolation=cv2.INTER_CUBIC)  # Downsample
            else:
                # Video mode
                ret_val, image = cap.read()
                
                # Skip frames to get realtime data representation
                if frame_skipped < SKIP_FRAME:
                    frame += 1
                    frame_skipped += 1
                    continue
                
                frame += 1
                frame_skipped = 0
            
            # Special smaller image for face recognition, reduces memory
            if len(imgs) == 5:
                imface = imf # Use face camera
            else:
                # Use cropped ceiling cams
                imface = image[freg[0]:freg[1], freg[2]:freg[3]]
            
            # Special masked image for openpose, reduce environment noise.
            # Draw a polygon mask around unwanted area, for 4 cam mode
            impose = image.copy()
            if DOMASK:
                for pmask in PMASK:
                    cv2.fillPoly(impose, [pmask], color=(200,200,288))
                    # cv2.fillPoly(impose, [pmask], color=(0,0,0))
            
            # Dummy image
            if DUMMY:
                impose[0:IMAGE[1], 0:IMAGE[0]] = rimg
                if (doff_x >= 0) and (doff_y >= 0) and (doff_x+dimg.shape[1] < IMAGE[0]) and (doff_y+dimg.shape[0] < IMAGE[1]):
                    impose[doff_y:doff_y+dimg.shape[0], doff_x:doff_x+dimg.shape[1]] = dimg
                    impose[doff_y+288:doff_y+dimg.shape[0]+288, 1024-(doff_x+dimg.shape[1]):1024-doff_x] = cv2.flip(dimg.copy(), 1)
                else:
                    doff_x = 0
                    doff_y = 30
                doff_x += int(round((1024-dimg.shape[1])/(3*4)))
                # doff_y += int(round((576-dimg.shape[0])/(3*4)))
            
            ###print("\n######################## Openpose")
            if SYS_OPOSE:
                human_keypoints, human_ids, humans = opose.runopenpose(impose)
                # print(humans, human_keypoints)
            else:
                human_keypoints = {0: [np.zeros(36)]}
                human_ids = {0: 0}
                humans = []
            
            ###print("\n######################## Darknet")
            if SYS_DARK:
                dobj = dark.performDetect(image)
                ###print(dobj)
            else:
                dobj = []
            
            ###print("\n######################## Facerec")
            if SYS_FACEREC:
                face_locs_tp, face_names_tp = facer.runinference(imface, tolerance=0.4, prescale=1/FPSCALE, upsample=FUP)
                ###print(face_locs_tp, face_names_tp)
            else:
                face_locs_tp = []
                face_names_tp = []
            
            # Prevent face blinking, apply the result if the new result is not empty.
            if face_locs_tp or hold_face <= 0:
                face_locs = face_locs_tp    # Apply the results
                face_names = face_names_tp
                hold_face = HFACE           # Reset counter
            else:
                hold_face -= 1
            
            # print("\n######################## LSTM")
            act_labs = []
            act_confs = []
            act_locs = []
            if SYS_ACT:
                for key, human_keypoint in human_keypoints.items():
                    ###print(key, human_keypoint)
                    if(len(human_keypoint)==N_STEPS):
                        act.runinference(human_keypoint)
                        act_labs.append(act.action)
                        act_confs.append(act.conf)
                        loc = openpose_human.average([human_keypoint[N_STEPS-1]])
                        # loc here is produced with format [[x,y]], so must be passing [0]
                        act_locs.append(loc[0])
            
            ###print("\n######################## Maths")
            sec_lv, sec_flv, sec_auths = self.sec_calc(sec_hist, image, act_labs, act_confs, human_keypoints, dobj, imface, face_names, face_locs, sec_auths)
            ###print(sec_lv)
            self.alert(sec_lv, len(sec_auths))
            
            ###print("\n######################## Display")
            # Main drawing procedure
            if DRAWMASK:
                # Draw openpose mask & face region
                self.display_all(impose, imface, sec_lv, sec_auths, humans, human_ids, act_labs, act_confs, act_locs, dobj, face_locs, face_names, freg)
            else:
                self.display_all(image, imface, sec_lv, sec_auths, humans, human_ids, act_labs, act_confs, act_locs, dobj, face_locs, face_names, freg)
            
            # Frame management stuffs, counted before frame limited
            frame_time = time.time() - ptime
            
            # FPS limiter
            if FPSLIM > 0:
                time.sleep(max(1./FPSLIM - (frame_time), 0))
                
            # FPS display & log, counted after frame limited
            self.fps = 1.0 / (time.time() - ptime)
            hisfps.append(self.fps)
            ptime = time.time()
            
            if cv2.waitKey(1) == 27:
                break
        
        cv2.destroyAllWindows()
        
        # Output FPS history
        fh = open("fps.txt", "w")
        for fps in hisfps:
            fh.write("%.3f \n" % fps)
        fh.close()
        
    def alert(self, sec_lv, sec_nauth):
        if self.altrig == 0: # From neutral
            # Alert & indicator about level below threshold
            if sec_lv < HISTH:
                winsound.PlaySound(None, winsound.SND_ASYNC)
                winsound.PlaySound(ALSND, winsound.SND_ASYNC | winsound.SND_ALIAS)
                self.altrig = 1 # To alert
                self.alprev = time.time()
            
        elif self.altrig == 1: # From alert
            if time.time() > self.alprev + ALDUR:
                self.altrig = 0 # To neutral
                winsound.PlaySound(None, winsound.SND_ASYNC)
        
        elif self.altrig == -1: # From cooldown period
            if time.time() > self.alprev + ALAUTH:
                self.altrig = 0 # To neutral

        elif self.altrig == -2: # From authorized
            # If none authorized
            if sec_nauth == 0:
                self.altrig = -1 # To cooldown period
                self.alprev = time.time()
                
        # Check authorization, nullify any security result if there's any authorized personnel
        if sec_nauth > 0:
            winsound.PlaySound(None, winsound.SND_ASYNC)
            self.altrig = -2
            
    def sec_calc(self, hist, image, act_labs, act_confs, human_keypoints, dobj, imface, face_names, face_locs, sec_auths, exth=EXTH):
        # Pass components used for security level calculations
        # TODO: implement threshold, constants, etc as variables
        sec = security.Frame(act_labs, act_confs, dobj, face_names)
        sec.calc()
        
        # Add to historical record
        # Base calculations from N latest data
        hist.append(sec)
        if (len(hist) > N_HIST):
            # Remove the last, only the view changed, no copy created
            hist.pop(0)
        all_hist = len(hist)
            
        # Calculation
        lvs = []
        for s in hist:
            lvs.append(s.level)
            print("%.3f " % s.level, end="")
        print("| | ", end ="")
        
        lvs = np.array(lvs)
        
        if all_hist >= N_HIST:
            if POSTPROC == 0:   # Count if
                sec_lv = len(lvs[lvs >= FRPARAM])/N_HIST
            elif POSTPROC == 1: # Average
                sec_lv = sum(lvs)/N_HIST
            elif POSTPROC == 2: # Percentile
                sec_lv = np.percentile(lvs, FRPARAM*100)
        else:
            sec_lv = 1.0
        
        # print("%d/%d %.2f | " % (all_neg, all_hist, sec_lv), end="")
        print("%.2f | " % (sec_lv), end="")
        
        # Print latest labels & confidence
        for act, conf in zip(act_labs, act_confs):
            print("%s[%.2f]," % (act, conf), end="")
        print()
        
        # Authorized exiting
        # Only check if there's no new face
        if len(sec_auths) > 0 and len(face_names) == 0:
            ##print(human_keypoints)
            for id, keys in human_keypoints.items(): # loc = (x,y)
                ###print(keys[-1], len(keys))
                
                # Get the last pose, only if the sequence is longer than 1 (has detected before)
                if len(keys) > 1:
                    pose = keys[-1] 
                    (x, y) = (int(pose[2]), int(pose[3]-5)) # pose[2],pose[3] = (x,y) of body center (chest)
                    if (EX[2] <= x <= EX[3]) and (EX[0] <= y <= EX[1]):
                        # Get surrounding colors, by radius EXR
                        ###print(loc[1]-EXR, loc[1]+EXR, loc[0]-EXR, loc[0]+EXR)
                        color = np.mean(image[y-EXR:y+EXR, x-EXR:x+EXR], axis=(0,1))
                        
                        frac = {}
                        # Check against every detected authorized
                        for auth in sec_auths:
                            (b1, g1, r1) = sec_auths[auth]
                            (b2, g2, r2) = color
                            dist = math.sqrt((b2-b1)**2+(g2-g1)**2+(r2-r1)**2)
                            
                            frac[auth] = (dist/441.67) # frac = dist/sqrt(255^2*3)
                            
                        # Get the one with smallest distance
                        minkey = min(frac, key=frac.get)
                        if frac[minkey] <= EXTH: # Check to threshold
                            sec_auths.pop(minkey)
                            
        # Authorization, just need one positive to trigger
        sec_flv = 0
        for name, (top, right, bottom, left) in zip(face_names, face_locs):
            if name != "Unknown":
                sec_flv += 1
                # Get color from the bottom row of imface
                color = np.mean(imface[-1,left:right].copy(), axis=0)
                sec_auths[name] = color # Designate that color to the person
        
        # Percentage
        return sec_lv, sec_flv, sec_auths
                            
        # Authorization, just need one positive to trigger
        sec_flv = 0
        for name, (top, right, bottom, left) in zip(face_names, face_locs):
            if name != "Unknown":
                sec_flv += 1
                # Get color from the bottom row of imface
                color = np.mean(imface[-1,left:right].copy(), axis=0)
                sec_auths[name] = color # Designate that color to the person
        
        # Percentage
        return sec_lv, sec_flv, sec_auths
        
    
    def display_all(self, image, imface, sec_lv, sec_auths, humans, human_ids, act_labs, act_confs, act_locs, objs, face_locs, face_names, freg=[]):
        # try:
        # from skimage import io, draw
        # import numpy as np
        # print("*** "+str(len(detections))+" Results, color coded by confidence ***")
        
        vt = 10 # Vertical positioning
        
        # Face camera display
        image[freg[0]:freg[1], freg[2]:freg[3]] = imface   # Insert to the center
        
        # Face region display
        if freg != []:
            cv2.rectangle(image, (freg[2], freg[0]), (freg[3], freg[1]), color=(64,64,64), thickness=1)
        
        # Exit region display
        cv2.rectangle(image, (EX[2], EX[0]), (EX[3], EX[1]), color=(64,64,64), thickness=1)
        
        # Openpose display
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        # Security level display
        color = (0, int(255 * sec_lv), int(255 * (1 - sec_lv)))
        cv2.rectangle(image, (10, vt), (self.im_w-10,vt+10), (255, 255, 255), thickness=1)
        cv2.rectangle(image, (10, vt), (int(round((self.im_w-20)*sec_lv)+10), vt+10), color, cv2.FILLED)
        cv2.rectangle(image, (int(round((self.im_w-20)*HISTH)+10-1), vt-5), (int(round((self.im_w-20)*HISTH)+10)+1,vt+10+5), (0, 0, 255), cv2.FILLED)
        vt += 30
        
        # Visual safety level indicator
        if self.altrig == 1: # Alert
            cv2.rectangle(image, (0, 0), (self.im_w, self.im_h), (0, 0, 255), thickness=8)
        elif self.altrig <= -1: # Authorized or cooldown
            cv2.rectangle(image, (0, 0), (self.im_w, self.im_h), (0, 255, 0), thickness=8)
        
        # Authorized names inside
        ht = 10    # For horizontal
        nvt = IMAGE[1]-10   # vt from bottom
        cv2.putText(image,
            "Auth: %2d |" % len(sec_auths),
            (ht, nvt),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)   
        ht += 85    # For horizontal
        for name in sec_auths:
            b, g, r = sec_auths[name]
            cv2.rectangle(image, (ht-2, nvt-15), (ht+15-2, nvt+4), (b,g,r), thickness=-1)
            cv2.rectangle(image, (ht-2, nvt-15), (ht+15-2, nvt+4), (255,255,255), thickness=1)
            cv2.putText(image,
                name[0], (ht, nvt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255-b, 255-g, 255-r), 1)   
            ht += 15
            
        # Extra stats
        cv2.putText(image,
            "SECURITY: %.0f%%" % (sec_lv*100),
            (10, vt),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)
        vt += 20
        
        cv2.putText(image,
            "FPS: %.2f" % self.fps,
            (10, vt),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 2)
        vt += 20
        
        # LSTM display
        for (act_lab, act_conf, act_loc, id_val) in zip(act_labs, act_confs, act_locs, human_ids.values()):
            ###print(act_lab, act_conf, act_loc, id_val)

            cv2.putText(image,
                "     %d: %s %.2f" % (id_val, act_lab, act_conf),
                (int(round(act_loc[0])), int(round(act_loc[1]))),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
            # vt += 20
        
        # Darknet display
        for obj in objs:
            ###print(obj)
            label = obj[0]
            dconf = obj[1]
            bounds = obj[2]
            
            image, color = openpose_human.draw_box(image, 1, bounds, label, dconf)
            
        # Facerec display
        for (top, right, bottom, left), face in zip(face_locs, face_names):
            ###print(face)
            label = face
            # bounds = [4*left, 4*top, 4*(right-left), 4*(bottom-top)]
            bounds = [freg[2]+FPSCALE*left, freg[0]+FPSCALE*top, FPSCALE*(right-left), FPSCALE*(bottom-top)]
            image, color = openpose_human.draw_box(image, 0, bounds, label, loc=1)
        
        cv2.imshow('Bedssys', image)


class openpose_human:
    # def __init__(self, camera=0,resize='0x0',resize_out_ratio=4.0,model='mobilenet_thin',show_process=False):
    def __init__(self, image, resize=OPSIZE, model='mobilenet_v2_small'):
        self.logger = logging.getLogger('TfPoseEstimator-WebCam')
        self.logger.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        ##self.logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
        self.w, self.h = model_wh(resize)
        if self.w > 0 and self.h > 0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=OPGPU)  # Allocate GPU fraction
            self.e = TfPoseEstimator(get_graph_path(model), target_size=(self.w, self.h), tf_config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
        ##self.logger.debug('cam read+')
        # cam = cv2.VideoCapture(camera)
        # ret_val, image = cam.read()
        self.im_h, self.im_w = image.shape[:2]
        # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
        self.videostep = 0
        self.human_keypoint = {0: [np.zeros(36)]}
        self.human_ids = {0: 0}
        
    def runopenpose(self, image, resize_out_ratio=4.0):
        # ret_val, image = cam.read()
        ##self.logger.debug('image process+')
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=resize_out_ratio)
        skeletoncount = 0
        skels = np.array([np.zeros(36)])
        
        for human in humans:
            if skeletoncount == 0:  # Initialize by adding N_STEPS of empty skeletons
                skels = np.array([openpose_human.write_coco_json(human, self.im_w,self.im_h)])
            else:                   # Append the rest
                skels = np.vstack([skels, np.array(openpose_human.write_coco_json(human, self.im_w,self.im_h))])
            skeletoncount = skeletoncount + 1
            
        # if skeletoncount == 1:  # Just assume it's the same prson if there's only one
            # self.human_keypoint[0].append(skels)
        
        if skeletoncount > 0:
            self.human_keypoint, self.human_ids = openpose_human.push(self.human_keypoint, self.human_ids, skels)
        else:
            # No human actually detected (humans is empty, thus skcount = 0)
            self.human_keypoint = {0: [np.zeros(36)]}
            self.human_ids = {0: 0}
        
        tf.reset_default_graph() # Reset the graph
        # self.logger.debug('finished+')
        
        return (self.human_keypoint, self.human_ids, humans)
        # Basically, human_keypoint store a string of poses, length N_STEPS, and tracked.
        # Humans is the result of a single inference, formatting still raw.
    
    def draw_box(image, coord_type, bounds, text='', conf=1, loc=0, thickness=3):
        # Based on the input detection coordinate
        if coord_type == 0:
            # Input (x, y) describes the top-left corner of detection
            x = int(bounds[0])
            y = int(bounds[1])
        else: # Input (x, y) describes the center of detection
            # Move it to the top-left corner
            x = int(bounds[0] - bounds[2]/2)
            y = int(bounds[1] - bounds[3]/2)
            
        w = int(bounds[2])
        h = int(bounds[3])
        
        color = (int(255 * (1 - (conf ** 2))), int(255 * (conf ** 2)), 0)
        
        # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        
        # Object text
        if loc == 0:
            cv2.putText(image, "%s %.2f" % (text, conf), (x, y-5),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        elif loc == 1:
            cv2.putText(image, "%s %.2f" % (text, conf), (x, y+h+15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image, color
    
    def write_coco_json(human, image_w, image_h):
        keypoints = []
        coco_ids = coco_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        for coco_id in coco_ids:
            if coco_id not in human.body_parts.keys():
                keypoints.extend([0, 0])
                continue
            body_part = human.body_parts[coco_id]
            keypoints.extend([round(body_part.x * image_w, 3), round(body_part.y * image_h, 3)])
        return keypoints

    def push(traces, ids, new_skels, THRESHOLD = 100, TRACE_SIZE = N_STEPS):
    
        ###print("##### Multi-human")
        
        """Add the keypoints from a new frame into the buffer."""
        # dists, neighbors = openpose_human.nearest_neighbors(traces, new_skels)
        dists, neighbors = openpose_human.point(traces, new_skels)
        keygen = []
        # New skeletons which aren't close to a previously observed skeleton:
        unslotted = []
        # Previously observed skeletons which aren't close to a new one:
        for each in traces.keys():
            keygen.append(each)
        unseen = set(keygen)
        for skel, dist, neighbor in zip(new_skels, dists, neighbors):
            ###print(dist, neighbor)
            if dist <= THRESHOLD:
                if neighbor in traces:
                    traces[neighbor].append(skel)
                else:
                    id = randint(0,100)     # Only used for naming
                    traces[neighbor] = []
                    traces[neighbor].append(skel)
                    ids[neighbor] = id
                if len(traces[neighbor]) > TRACE_SIZE:
                    traces[neighbor].pop(0)
                unseen.discard(neighbor)
            else:
                unslotted.append(skel)

        for i in unseen:
            del traces[i]
            del ids[i]

        # Indices we didn't match, and the rest of the numbers are fair game
        availible_slots = chain(sorted(unseen), count(len(traces)))
        for slot, skel in zip(availible_slots, unslotted):
            id = randint(0,100)     # Only used for naming
            if slot in traces:
                traces[slot].append(skel)
            else:
                traces[slot] = []
                traces[slot].append(skel)
                ids[slot] = id
                
        return traces, ids
    
    def point(traces, skels, TRACE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]):
        if not traces:  # First pass
            return np.zeros(len(skels)), np.arange(len(skels))
            
        prev = np.array([  # Pull the most recent location of each skeleton, [-1] means get 1 data from behind
            coords[-1][TRACE_IDX] for _, coords in sorted(traces.items())])
            
        curr = skels[:, TRACE_IDX]
        # Determine representative point, may use various method such as median, average, etc
        prev_point = openpose_human.average(prev)
        curr_point = openpose_human.average(curr)
        
        # N is typically small (< 40) so brute force is fast
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='brute')
        nn_model.fit(prev_point)
        dist, nn = nn_model.kneighbors(curr_point, return_distance=True)
        
        return dist.flatten(), nn.flatten()
    
    def average(skels):
        avg_skels = np.empty((0, 2))
        for skel in skels:
            # Remember that a point might not be detected, giving zero. Count the non-zero.
            # Below line is equivalent to COUNTIF(not-zero).
            
            # Count non-zeros
            nzero_x = sum(1 if (x != 0) else 0 for x in skel[SKX])
            nzero_y = sum(1 if (x != 0) else 0 for x in skel[SKY])
            
            if (nzero_x == 0):
                nzero_x = 1
            if (nzero_y == 0):
                nzero_y = 1
                
            x = sum(skel[SKX]) / nzero_x
            y = sum(skel[SKY]) / nzero_y
            avg_skels = np.vstack((avg_skels, np.array([x, y])))
            
        return avg_skels
    
    def nearest_neighbors(traces, skels, TRACE_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]):
    
        if not traces:  # First pass
            return np.zeros(len(skels)), np.arange(len(skels))
        prev = np.array([  # Pull the most recent location of each skeleton
            coords[-1][TRACE_IDX] for _, coords in sorted(traces.items())])
        curr = skels[:, TRACE_IDX]
        # N is typically small (< 40) so brute force is fast
        nn_model = NearestNeighbors(n_neighbors=1, algorithm='brute')
        nn_model.fit(prev)
        dist, nn = nn_model.kneighbors(curr, return_distance=True)
        return dist.flatten(), nn.flatten()
 
        
class activity_human:

    action = "null"
    conf = 0
    loc = []
    
    # LABELS = [    
        # "JUMPING",
        # "JUMPING_JACKS",
        # # "BOXING",
        # "WAVING_2HANDS",
        # "WAVING_1HAND",
        # "CLAPPING_HANDS"
    # ] 
        
    def __init__(self):
        self.LABELS = LABELS
        self.n_input = 36
        
        self.n_hidden = 36 # Hidden layer num of features
        # n_classes = 6
        n_classes = len(self.LABELS)
        # N_STEPS = 32
        
        #updated for learning-rate decay
        # calculated as: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
        decaying_learning_rate = True
        learning_rate = 0.0025 #used if decaying_learning_rate set to False
        init_learning_rate = 0.005
        decay_rate = 0.96 #the base of the exponential in the decay
        decay_steps = 100000 #used in decay every 60000 steps with a base of 0.96
        global_step = tf.Variable(0, trainable=False)
        lambda_loss_amount = 0.0015
        # training_iters = training_data_count *300  # Loop 300 times on the dataset, ie 300 epochs
        # training_iters = training_data_count *60
        # training_iters = training_data_count *120
        # training_iters = training_data_count *1
        batch_size = 512
        display_iter = batch_size*8  # To show test set accuracy during training
             
        #### Build the network
        # Graph input/output
        self.x = tf.placeholder(tf.float32, [None, N_STEPS, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, n_classes])
        # Graph weights
        weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_input, self.n_hidden])), # Hidden layer weights
            'out': tf.Variable(tf.random_normal([self.n_hidden, n_classes], mean=1.0))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        self.pred = activity_human.LSTM_RNN(self, self.x, weights, biases)
        
        # Loss, optimizer and evaluation
        l2 = lambda_loss_amount * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        ) # L2 loss prevents this overkill neural network to overfit the data
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.pred)) + l2 # Softmax loss
        
        if decaying_learning_rate:
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)
            
        #decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) #exponentially decayed learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost,global_step=global_step) # Adam Optimizer
        
        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=LSGPU)  # Allocate GPU fraction
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))
        # self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        
        # self.sess = tf.self.session(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        # training_iters = training_data_count *30
        #create saver before training
        saver = tf.train.Saver()
        # saver = tf.train.Saver(var_list={'wh':weights['hidden'], 'wo':weights['out'], 'bh':biases['hidden'], 'bo':biases['out']})
        
        # tf.reset_default_graph()
        
        load = True
        train = False
        update = False
        #check if you want to retrain or import a saved model
        
        if load:
            saver.restore(self.sess, DATASET_PATH + "model.ckpt")
            ###print("Model restored.")
        
        correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
    # Load the networks inputs
    def runinference(self, human_keypoint):
        time_start = time.time()    
        ##### Inferencing
        # X_infer_path = "utilities/something/something.txt"
        # X_infer_path = DATASET_PATH + "X_test.txt"
        # X_val = load_X(X_infer_path)
        X_test = activity_human.load_XLive(human_keypoint)
        
        # print("##### Raw")
        # for arr in human_keypoint:
            # for i in arr:
                # print(i, end=", ")
            # print()
            
        
        # print("##### Preprocessed")
        # if len(X_test) > 0:
            # for arr in X_test[0]:
                # for i in arr:
                    # print(i, end=", ")
                # print()
        
        self.preds = self.sess.run(
            [self.pred],
            feed_dict={
                self.x: X_test
           }
        )
        
        # Special selection (temporary confidence)
        tconf = self.preds[0][0].copy()
        
        if LABSEL[0]:
            # Weighted
            tconf *= LABWEI
        
        if LABSEL[1]:
            # Grouped
            avgs = [sum(tconf[gr])/len(gr) for gr in LABGRO]
            imax = avgs.index(max(avgs))
            
            # The losing group is nullified
            zeros = np.zeros(len(tconf))
            zeros[LABGRO[imax]] = 1
            
            tconf *= zeros
        
        # Basic selection
        id, self.conf = max(enumerate(tconf), key=operator.itemgetter(1))
        
        self.action = self.LABELS[id]
        
        ###print(tconf)
        ###print(self.preds, self.action)
        
        time_stop = time.time()
        ###print("TOTAL TIME:  {}".format(time_stop - time_start))
        
    
        
    def load_X(X_path):
        file = open(X_path, 'r')
        X_ = np.array(
            [elem for elem in [
                row.split(',') for row in file
            ]], 
            dtype=np.float32
        )
        file.close()
        blocks = int(len(X_) / N_STEPS)
        X_ = np.array(np.split(X_,blocks))
        return X_ 

    # Load the networks inputs
    def load_XLive(keypoints):
        # print(keypoints)
        # print(len(keypoints), ":", [len(row) for row in keypoints])
        
        X_ = np.array(keypoints,dtype=np.float32)
        
        blocks = int(len(X_) / N_STEPS)
        X_ = np.array(np.split(X_,blocks))
        
        # Idle check & forcing it if it is.
        if PREPROC[1] == 1:
            X_[0] = activity_human.idlenull(X_[0])
        elif PREPROC[1] == 2:
            X_[0] = activity_human.idlenull2(X_[0])
        
        # Preprocessing before the data is used for inference
        # The data is: [ [ [point x 36] x N_STEPS] ], so one too many layer
        if PREPROC[0] == 1:
            # Poses emulated as if there's a big border between sub-images
            X_[0] = activity_human.amplify(X_[0])
        elif PREPROC[0] == 2:
            # Individual pose returned to origin
            X_[0] = activity_human.normalize(X_[0])
        elif PREPROC[0] == 3:
            # Every pose in a gesture will be relative to the first in the gesture
            X_[0] = activity_human.normalizeonce(X_[0])
        elif PREPROC[0] == 4:
            # Every pose in a gesture will be relative to the first in the gesture
            X_[0] = activity_human.normalizepoint(X_[0])
        elif PREPROC[0] == 5:
            # Poses in 4 sub-images emulated as if happening in a single image
            X_[0] = activity_human.reverse(X_[0])
            
        return X_ 
    
    def idlenull2(skels):
        # Preprocess, force any unmoving gesture to be idle
        diff_x = 0
        diff_y = 0
        n = 5
        for i, skel in enumerate(skels):
            # Calculate the midpoint representation,
            # using primary points.
            
            # Try pose part #1, neck
            ax = skel[2]
            ay = skel[3]
            
            if ax==0 and ay==0:
                # Then try pose part #0, nose
                ax = skel[0]
                ay = skel[1]
            
            # Calculate then sum overall movement
            if i != 0:
                diff_x += abs(ax - px)
                diff_y += abs(ay - py)
            
            px = ax
            py = ay
            
        # Average the diff and calculate the distance
        diff_x /= n-1
        diff_y /= n-1
        diff = math.sqrt(diff_x**2 + diff_y**2)
        
        if diff < IDLETH:
            # All to zero, tested that it's guaranteed to be inferenced as idle (tho low confidence).
            skels = np.array(N_STEPS * [[478,62,476,78,492,80,494,108,494,132,458,76,442,100,440,128,478,128,474,158,476,188,454,126,442,158,426,194,480,60,476,60,484,62,474,60]], dtype=np.float32)
            
        return skels
    
    def idlenull(skels):
        # Preprocess, force any unmoving gesture to be idle
        diff_x = 0
        diff_y = 0
        n = 5
        for i, skel in enumerate(skels):
            # Calculate the midpoint representation, using average
        
            # (Exact copy from average function)
            # Remember that a point might not be detected, giving zero. Count the non-zero.
            # Below line is equivalent to COUNTIF(not-zero).
            x = skel[SKX]
            y = skel[SKY]
            
            # Count non-zeros
            nzero_x = sum(1 if (k != 0) else 0 for k in x)
            nzero_y = sum(1 if (k != 0) else 0 for k in y)
            
            if (nzero_x == 0 and nzero_y == 0):
                n -= 1
            if (nzero_x == 0):
                nzero_x = 1
            if (nzero_y == 0):
                nzero_y = 1
                
            ax = sum(x) / nzero_x
            ay = sum(y) / nzero_y
            
            # Calculate then sum overall movement
            if i != 0:
                diff_x += abs(ax - px)
                diff_y += abs(ay - py)
            
            px = ax
            py = ay
            
        # Average the diff and calculate the distance
        diff_x /= n-1
        diff_y /= n-1
        diff = math.sqrt(diff_x**2 + diff_y**2)
        
        if diff < IDLETH:
            # All to zero, tested that it's guaranteed to be inferenced as idle (tho low confidence).
            skels = np.array(N_STEPS * [[478,62,476,78,492,80,494,108,494,132,458,76,442,100,440,128,478,128,474,158,476,188,454,126,442,158,426,194,480,60,476,60,484,62,474,60]], dtype=np.float32)
            
        return skels
    
    def normalizepoint(skels):
        # Preprocess, move any pose to the origin, based on their average as midpoint ref.
        # Still using old normalization method instead of the one used in normalizeonce.
        for i, skel in enumerate(skels):
            # Calculate the midpoint representation, using average
            
            x = skel[SKX]
            y = skel[SKY]
            
            if (i == 0):
                xo = x.copy()
                yo = y.copy()
            
            # Normalization process
            # Shifting first pose to origin, and the rest follow the same shift
            zero = [0 if (k == 0) else 1 for k in skel] # As the multiplier, zero stays zero
            
            x -= xo
            y -= yo
            
            # Recombine, placed one after another
            skel[0::2] = x
            skel[1::2] = y
            
            skel *= zero
        return skels
    
    def normalizeonce(skels):
        first = False
        
        # Preprocess, move any pose to the origin, based on their average as midpoint ref.
        for skel in skels:
            # Calculate the midpoint representation, using average
            # Similar copy from average function

            x = skel[SKX]
            y = skel[SKY]
            
            if (first == False):
                # Remember that a point might not be detected, giving zero.
                # Count the non-zero.
                nzero_x = sum(1 if (k != 0) else 0 for k in x)
                nzero_y = sum(1 if (k != 0) else 0 for k in y)
                
                if (nzero_x == 0) and (nzero_y == 0):
                    first = False
                else:
                    first = True
                    
                    if (nzero_x == 0):
                        nzero_x = 1
                    if (nzero_y == 0):
                        nzero_y = 1
                    
                    ax = sum(x) / nzero_x
                    ay = sum(y) / nzero_y
                    
            if (first == True):
                # Normalization process
                # Shifting first pose to origin, and the rest follow the same shift
                zero = [0 if (k == 0) else 1 for k in skel] # As the multiplier, zero stays zero
                
                x -= ax
                y -= ay
                
                # Recombine, placed one after another
                skel[0::2] = x
                skel[1::2] = y
            
                skel *= zero
        return skels
        
    def normalize(skels):
        # Preprocess, move any pose to the origin, based on their average as midpoint ref.
        for skel in skels:
            # Calculate the midpoint representation, using average
            # Similar copy from average function
            x = skel[SKX]
            y = skel[SKY]
            
            # Remember that a point might not be detected, giving zero.
            # Count the non-zero.
            nzero_x = sum(1 if (k != 0) else 0 for k in x)
            nzero_y = sum(1 if (k != 0) else 0 for k in y)
            
            if (nzero_x == 0):
                nzero_x = 1
            if (nzero_y == 0):
                nzero_y = 1
            
            ax = sum(x) / nzero_x
            ay = sum(y) / nzero_y
            
            # Normalization process
            # Shifting every poses to origin
            zero = [0 if (k == 0) else 1 for k in skel] # As the multiplier, zero stays zero
            
            x -= ax
            y -= ay
            
            # Recombine, placed one after another
            skel[0::2] = x
            skel[1::2] = y
            
            skel *= zero
        return skels
    
    def amplify(skels):
        # Preprocess, move any pose in different quadrant way further from each other.
        for skel in skels:
            # Similar copy from average function
            
            x = skel[SKX]
            y = skel[SKY]
            
            # Remember that a point might not be detected, giving zero.
            # Count the non-zero.
            nzero_x = sum(1 if (k != 0) else 0 for k in x)
            nzero_y = sum(1 if (k != 0) else 0 for k in y)
            
            if (nzero_x == 0):
                nzero_x = 1
            if (nzero_y == 0):
                nzero_y = 1
            
            ax = sum(x) / nzero_x
            ay = sum(y) / nzero_y
            
            # Amplification process
            # Shifting constant, to be added to the skeletons
            sx = POSEAMP if (ax > SUBIM[0]) else 0
            sy = POSEAMP if (ay > SUBIM[1]) else 0
                
            zero = [0 if (x == 0) else 1 for x in skel] # As the multiplier, zero stays zero
            
            x += sx
            y += sy
            
            # Recombine, placed one after another
            skel[0::2] = x
            skel[1::2] = y
            
            skel *= zero
        return skels
        
    def reverse(skels):
        # Reverse of amplify
        # Preprocess, move any pose in different quadrant as if it's happening in single quadrant.
        for skel in skels:
            # Similar copy from average function
            
            x = skel[SKX]
            y = skel[SKY]
            
            # Remember that a point might not be detected, giving zero.
            # Count the non-zero.
            nzero_x = sum(1 if (k != 0) else 0 for k in x)
            nzero_y = sum(1 if (k != 0) else 0 for k in y)
            
            if (nzero_x == 0):
                nzero_x = 1
            if (nzero_y == 0):
                nzero_y = 1
            
            ax = sum(x) / nzero_x
            ay = sum(y) / nzero_y
            
            # Amplification process
            # Shifting constant, to be added to the skeletons
            sx = SUBIM[0] if (ax >= SUBIM[0]) else 0
            sy = SUBIM[1] if (ay >= SUBIM[1]) else 0
                
            zero = [0 if (x == 0) else 1 for x in skel] # As the multiplier, zero stays zero
            
            x -= sx
            y -= sy
            
            # Recombine, placed one after another
            skel[0::2] = x
            skel[1::2] = y
            
            skel *= zero
        return skels

    def load_y(y_path):
        file = open(y_path, 'r')
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]], 
            dtype=np.int32
        )
        file.close()
        
        # for 0-based indexing 
        return y_ - 1

    def LSTM_RNN(self, _X, _weights, _biases):
        # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

        _X = tf.transpose(_X, [1, 0, 2])  # permute N_STEPS and batch_size
        _X = tf.reshape(_X, [-1, self.n_input])   
        # Rectifies Linear Unit activation function used
        _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, N_STEPS, 0) 

        if LAYER == 1:
            # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
            lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        
        elif LAYER == 2:
            # Single hidden layer
            lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1], state_is_tuple=True)
        
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

        # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
        lstm_last_output = outputs[-1]
        
        # Linear activation
        return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


    def extract_batch_size(_train, _labels, _unsampled, batch_size):
        # Fetch a "batch_size" amount of data and labels from "(X|y)_train" data. 
        # Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label from Y_train
        # unsampled_indices keeps track of sampled data ensuring non-replacement. Resets when remaining datapoints < batch_size    
        
        shape = list(_train.shape)
        shape[0] = batch_size
        batch_s = np.empty(shape)
        batch_labels = np.empty((batch_size,1)) 

        for i in range(batch_size):
            # Loop index
            # index = random sample from _unsampled (indices)
            index = random.choice(_unsampled)
            batch_s[i] = _train[index] 
            batch_labels[i] = _labels[index]
            
            _unsampled = list(_unsampled)
            
            _unsampled.remove(index)

        return batch_s, batch_labels, _unsampled

    def one_hot(y_):
        # One hot encoding of the network outputs
        # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
        y_ = y_.reshape(len(y_))
        n_values = int(np.max(y_)) + 1
        return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
        
    def test(self):
        X_train_path = DATASET_PATH + "X_train.txt"
        X_test_path = DATASET_PATH + "X_test.txt"
        y_train_path = DATASET_PATH + "Y_train.txt"
        y_test_path = DATASET_PATH + "Y_test.txt"

        X_train = activity_human.load_X(X_train_path)
        X_test = activity_human.load_X(X_test_path)
        y_train = activity_human.load_y(y_train_path)
        y_test = activity_human.load_y(y_test_path)
        
        # only perform testing - on training set
        loss, acc = self.sess.run(
            [self.cost, self.accuracy], 
            feed_dict={
                self.x: X_train,
                self.y: activity_human.one_hot(y_train)
            }
        )
        
        print()
        print("PERFORMANCE ON TRAIN SET:             " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))
        
        # only perform testing - on test set
        loss, acc = self.sess.run(
            [self.cost, self.accuracy], 
            feed_dict={
                self.x: X_test,
                self.y: activity_human.one_hot(y_test)
            }
        )
        
        print("PERFORMANCE ON TEST SET:             " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))
        print()
     
if __name__ == '__main__':
    mainhuman_activity()

