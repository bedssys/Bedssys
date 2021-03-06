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
import struct
import operator
from imutils.video import WebcamVideoStream
import imutils

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from itertools import chain, count
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

import darknet.json as dk
import facerec.recognize as fr
# import deepface.deepface as df

n_steps = 5
# DATASET_PATH = "data/"
# DATASET_PATH = "data/Overlap_fixed/"
DATASET_PATH = "data/Overlap_fixed4/"
# DATASET_PATH = "data/Overlap_fixed4_separated/"

# <TODO> camera url
# 4 camera
# mode openpose dan kamera
# simpan gambar dan simpan mp4
# face tolerance
# alarm activation tombol
# setting security for type data threshold
LABELS = [
        "GO_IN",
        "GO_OUT",
        "WALK_LEFT",
        "WALK_RIGHT"
    ]

# CAMERA = [0, 2]
# CAMERA = [0]
CAMERA = ["rtsp://167.205.66.147:554/onvif1", "rtsp://167.205.66.148:554/onvif1", "rtsp://167.205.66.149:554/onvif1",  "rtsp://167.205.66.150:554/onvif1"]
# ROTATE = [0, 0, 0, 0]
ROTATE = [180, 180, 180, 180]
class mainhuman_activity:

    # Pre-processing for every image
    def preprocess(raws, rots):
        imgs = []
        for img, rot in zip(raws, rots):
            # img = cv2.resize(img, dsize=(256, 144), interpolation=cv2.INTER_CUBIC)    # 16:9
            img = cv2.resize(img, dsize=(512, 288), interpolation=cv2.INTER_CUBIC)  # 16:9
            # img = cv2.resize(img, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)  # 4:3
            # img = cv2.resize(img, dsize=(320, 240), interpolation=cv2.INTER_CUBIC)  # 4:3
            # img = cv2.resize(img, dsize=(160, 120), interpolation=cv2.INTER_CUBIC)  # 4:3
            # img = imutils.rotate_bound(img, 90)
            img = imutils.rotate_bound(img, rot)
            imgs.append(img)
            
        if len(imgs) == 1:
            image = imgs[0]
        if len(imgs) >= 2:
            image = np.hstack((imgs[0], imgs[1]))
        if len(imgs) == 4:
            image2 = np.hstack((imgs[2], imgs[3]))
            image = np.vstack((image, image2))
            
        return image
    
    def __init__(self, camera=CAMERA, rotate = ROTATE):
        cams = [WebcamVideoStream(src=cam).start() for cam in camera]
        
        imgs = []
        for i, cam in enumerate(cams):
            # cam.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Internal buffer will now store only x frames
            img = cam.read()
            imgs.append(img)
            
        image = mainhuman_activity.preprocess(imgs, rotate)
        
        # h, w, c = image_raw.shape
        # h2, w2, c2 = image2_raw.shape
        
        # print(h, w, c, h2, w2, c2)
        
        print("\n######################## Darknet")
        dark = dk.darknet_recog()
        print(dark.performDetect(image))
        
        print("\n######################## Openpose")
        opose = openpose_human(image)
        
        print("\n######################## LSTM")
        act = activity_human()
        
        # print("\n######################## Deepface")
        # dface = df.face_recog()
        # print(dface.run(image))
        
        print("\n######################## Facerec")
        facer = fr.face_recog(face_dir="./facerec/face/")
        
        act_labs = []
        act_confs = []
        
        # Main loop
        try:
            f = open(r'\\.\pipe\testing', 'r+b',0)
            d = 0 # mode in communication
            alarmmode = False # False mode deactive True mode active
            mode = True # False normal mode True recognition mode
            security_threshold = 0.5
            face_tolerance = 0.6
            while True:
                # imgs = [mainhuman_activity.read2(cam) for cam in cams]
                n = struct.unpack('I', f.read(4))[0]    # Read str length
                s = f.read(n).decode('ascii')           # Read str
                f.seek(0)
                print ('Accept from C#', s)
                if (s == 'AlarmDeactive'):
                    d = 7
                elif (s == 'AlarmActive'):
                    d = 6
                elif (s == 'FaceInput'):
                    d = 5
                elif (s == 'Normal'):
                    d = 4
                elif (s == 'Recognition'):
                    d = 3
                elif (s == 'Start'):
                    d = 2
                elif (s == 'Stop'):
                    d = 1
                elif (s == 'Received'):
                    d = 0
                else:
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                if (d == 7):
                    alarmmode = False # False mode deactive True mode active
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                elif (d == 6):
                    alarmmode = True # False mode deactive True mode active
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                elif (d == 5):
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                    n = struct.unpack('I', f.read(4))[0]    # Read str length
                    facename = f.read(n).decode('ascii')           # Read str
                    f.seek(0)
                    print ('Accept from C#', facename)
                    imgs = []
                    
                    img = cams[0].read()
                    imgs.append(img)
                        
                    
                    # for i, cam in enumerate(cams):
                        # # Decode the captured frames
                        # ret_val, img = cam.retrieve()
                        # imgs.append(img)
                    
                    # Skip frame if there's nothing
                    if(imgs is [None]):
                        continue
                        
                    image = mainhuman_activity.preprocess(imgs, rotate)
                    face_locs, face_names = facer.runinference(image, tolerance=face_tolerance, prescale=0.25, upsample=2)
                    # Facerec display
                    for (top, right, bottom, left), face in zip(face_locs, face_names):
                        print(face)
                        if (face == "Unknown"):
                            bounds = [4*left, 4*top, 4*right, 4*bottom]
                            image = image[bounds[1]:bounds[3], bounds[0]:bounds[2]]
                    cv2.imwrite('facerec/face/'+facename+'.jpg', image)
                    print("\n######################## Facerec")
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                elif (d == 4):
                    mode = False # False normal mode True recognition mode
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                elif (d == 3):
                    mode = True # False normal mode True recognition mode
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                elif (d == 2):
                    for i, cam in enumerate(cams):
                        # cam.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Internal buffer will now store only x frames
                        cam.stop()
                    camera = []
                    rotate = []
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                    n = struct.unpack('I', f.read(4))[0]    # Read str length
                    camnumber = f.read(n).decode('ascii')           # Read str
                    f.seek(0)
                    try:
                        cam_number = int(camnumber)
                    except ValueError:
                        pass
                    print ('Accept from C#', camnumber)
                    for x in range(cam_number):
                        s='Go'
                        f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                        f.seek(0)
                        print ('Sending to C#:', s)
                        n = struct.unpack('I', f.read(4))[0]    # Read str length
                        camtemp = f.read(n).decode('ascii')           # Read str
                        f.seek(0)
                        try:
                            camera.append(int(camtemp))
                            rotate.append(180)
                        except ValueError:
                            camera.append(camtemp)
                            rotate.append(180)
                            pass
                        print ('Accept from C#', camtemp)
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                    n = struct.unpack('I', f.read(4))[0]    # Read str length
                    securitythresholdtemp = f.read(n).decode('ascii')           # Read str
                    f.seek(0)
                    if (securitythresholdtemp!=" "):
                        try:
                            security_threshold = float(securitythresholdtemp)
                        except ValueError:
                            pass
                    print ('Accept from C#', securitythresholdtemp)
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                    n = struct.unpack('I', f.read(4))[0]    # Read str length
                    facetolerancetemp = f.read(n).decode('ascii')           # Read str
                    f.seek(0)
                    if (facetolerancetemp!=" "):
                        try:
                            face_tolerance = float(facetolerancetemp)
                        except ValueError:
                            pass
                    print ('Accept from C#', facetolerancetemp)
                    cams = [WebcamVideoStream(src=cam).start() for cam in camera]
                    s='Go'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                elif(d == 1):
                    s='Wait'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                elif(d == 0):
                    imgs = []
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
                        imgs.append(img)
                        
                    
                    # for i, cam in enumerate(cams):
                        # # Decode the captured frames
                        # ret_val, img = cam.retrieve()
                        # imgs.append(img)
                    
                    # Skip frame if there's nothing
                    if(imgs is [None]):
                        continue
                        
                    image = mainhuman_activity.preprocess(imgs, rotate)
            
                    print("\n######################## Openpose")
                    start_act, human_keypoints, humans = opose.runopenpose(image)
                    # print(humans, human_keypoints)
                    
                    print("\n######################## Darknet")
                    dobj = dark.performDetect(image)
                    print(dobj)
                    
                    print("\n######################## Facerec")
                    face_locs, face_names = facer.runinference(image, tolerance=face_tolerance, prescale=0.01, upsample=1)
                    print(face_locs, face_names)
                    
                    print("\n######################## LSTM")
                    print("Frame: %d/%d" % (opose.videostep, n_steps))
                    if start_act == True:
                        act_labs = []
                        act_confs = []
                        for key, human_keypoint in human_keypoints.items():
                            print(key, human_keypoint)
                            if(len(human_keypoint)==n_steps):
                                act.runinference(human_keypoint)
                                act_labs.append(act.action)
                                act_confs.append(act.conf)
                                
                    print("\n######################## Display")
                    # opose.display_all(image, humans, act.action, act.conf, dobj, face_locs, face_names)
                    opose.display_all(image, humans, act_labs, act_confs, dobj, face_locs, face_names, mode)
                    s='Image'
                    f.write(struct.pack('I', len(s)) + s.encode('ascii'))   # Write str length and str
                    f.seek(0)
                    print ('Sending to C#:', s)
                
                if cv2.waitKey(1) == 27:
                    break
                 
        except FileNotFoundError :
            raise
        cv2.destroyAllWindows()
        
        # print("FPS: ", opose.hisfps)
        fh = open("fps.txt", "w")
        for fps in opose.hisfps:
            fh.write("%.3f \n" % fps)
        fh.close()


class openpose_human:
    # def __init__(self, camera=0,resize='0x0',resize_out_ratio=4.0,model='mobilenet_thin',show_process=False):
    def __init__(self, image, resize='0x0',model='mobilenet_thin'):
        self.logger = logging.getLogger('TfPoseEstimator-WebCam')
        self.logger.setLevel(logging.DEBUG)
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        self.ch.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        self.logger.debug('initialization %s : %s' % (model, get_graph_path(model)))
        self.w, self.h = model_wh(resize)
        if self.w > 0 and self.h > 0:
            self.e = TfPoseEstimator(get_graph_path(model), target_size=(self.w, self.h))
        else:
            self.e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
        self.logger.debug('cam read+')
        # cam = cv2.VideoCapture(camera)
        # ret_val, image = cam.read()
        self.image_h, self.image_w = image.shape[:2]
        # logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
        self.fps_time = 0
        self.videostep = 0
        self.human_keypoint = {0: [np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])]}
        self.hisfps = []        # Historical FPS data
        
    def runopenpose(self, image, resize_out_ratio=4.0):
        # ret_val, image = cam.read()
        self.logger.debug('image process+')
        humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=resize_out_ratio)
        skeletoncount = 0
        skels = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        
        for human in humans:
            if skeletoncount == 0:  # Initialize
                skels = np.array([openpose_human.write_coco_json(human, self.image_w,self.image_h)])
            else:                   # Append the rest
                skels = np.vstack([skels, np.array(openpose_human.write_coco_json(human, self.image_w,self.image_h))])
            skeletoncount = skeletoncount + 1
            
        # if skeletoncount == 1:  # Just assume it's the same prson if there's only one
            # self.human_keypoint[0].append(skels)
        if skeletoncount > 0:
            self.human_keypoint = openpose_human.push(self.human_keypoint, skels)

        self.videostep += 1
        if (self.videostep == n_steps):
            start_act = True
            human_keypointer = self.human_keypoint
            self.videostep = 0
        else:
            start_act = False
            human_keypointer = {}
        
        tf.reset_default_graph() # Reset the graph
        # self.logger.debug('finished+')
        return(start_act, human_keypointer, humans)
        
    # def runopenpose(self, image, resize_out_ratio=4.0):
        # # ret_val, image = cam.read()
        # self.logger.debug('image process+')
        # humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=resize_out_ratio)
        # skeletoncount = 0
        # skels = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        # for human in humans:
            # if skeletoncount == 0:
                # skels = np.array([openpose_human.write_coco_json(human,self.image_w,self.image_h)])
            # else:
                # skels = np.vstack([skels, np.array(openpose_human.write_coco_json(human,self.image_w,self.image_h))])
            # skeletoncount = skeletoncount + 1
        # if skeletoncount > 0:
            # self.human_keypoint = openpose_human.push(self.human_keypoint,skels)
        # # if humans:
            # # self.human_keypoint.append(openpose_human.write_coco_json(humans[0],self.image_w,self.image_h))
        # # else:
            # # self.human_keypoint.append([0 for x in range(0,36)])
        # self.videostep += 1
        # if (self.videostep == n_steps):
            # start_act = True
            # human_keypointer = self.human_keypoint
            # self.videostep = 0
        # else:
            # start_act = False
            # human_keypointer = {}
        
        # tf.reset_default_graph() # Reset the graph
        # # self.logger.debug('finished+')
        # return(start_act, human_keypointer, humans)
        
    def display_all(self, image, humans, act_labs, act_confs, detections, face_locs, face_names, mode):
        # try:
        # from skimage import io, draw
        # import numpy as np
        # print("*** "+str(len(detections))+" Results, color coded by confidence ***")
        if(mode):
            vt = 10
            
            # Openpose & LSTM display
            self.logger.debug('postprocess+')
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            self.logger.debug('show+')
            
            fps = 1.0 / (time.time() - self.fps_time)
            self.hisfps.append(fps)
            
            cv2.rectangle(image, (10, vt), (self.image_w-10,vt+10), (0, 128, 0), cv2.FILLED)
            cv2.rectangle(image, (10, vt), (10+round((self.image_w-10)*self.videostep/n_steps),vt+10), (0, 255, 0), cv2.FILLED)
            vt += 30
            
            cv2.putText(image,
                "FPS: %f" % fps,
                (10, vt),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
            vt += 20

            for (act_lab, act_conf) in zip(act_labs, act_confs):
                cv2.putText(image,
                    "PRED: %s %.2f" % (act_lab, act_conf),
                    (10, vt),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
                vt += 20
            
            # Darknet display
            for detection in detections:
                print(detection)
                label = detection[0]
                dconf = detection[1]
                bounds = detection[2]
                
                image, color = openpose_human.draw_box(image, 1, bounds, label, dconf)
                
            # Facerec display
            for (top, right, bottom, left), face in zip(face_locs, face_names):
                print(face)
                label = face
                bounds = [4*left, 4*top, 4*(right-left), 4*(bottom-top)]
                image, color = openpose_human.draw_box(image, 0, bounds, label, loc=1)
            
            cv2.imwrite('./IPC CS/bin/Release/display_sharp.jpg', image)
                
            self.fps_time = time.time()
            self.logger.debug('finished+')
        else:
            cv2.imwrite('./IPC CS/bin/Release/display_sharp.jpg', image)
    
    def draw_box(image, coord_type, bounds, text='', conf=1, loc=0):
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
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
        
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

    def push(traces, new_skels, THRESHOLD = 100, TRACE_SIZE = n_steps):
    
        print("##### Multi-human")
        
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
            print(dist, neighbor)
            if dist <= THRESHOLD:
                if neighbor in traces:
                    traces[neighbor].append(skel)
                else:
                    traces[neighbor] = []
                    traces[neighbor].append(skel)
                if len(traces[neighbor]) > TRACE_SIZE:
                    traces[neighbor].pop(0)
                unseen.discard(neighbor)
            else:
                unslotted.append(skel)

        for i in unseen:
            del traces[i]

        # Indices we didn't match, and the rest of the numbers are fair game
        availible_slots = chain(sorted(unseen), count(len(traces)))
        for slot, skel in zip(availible_slots, unslotted):
            if slot in traces:
                traces[slot].append(skel)
            else:
                traces[slot] = []
                traces[slot].append(skel)
        return traces
    
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
            x = sum(skel[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]]) / 18
            y = sum(skel[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]]) / 18
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
        # Useful Constants
        # Output classes to learn how to classify
        # DATASET_PATH = "data/HAR_pose_activities/database/"
        # X_train_path = DATASET_PATH + "X_train.txt"
        # X_test_path = DATASET_PATH + "X_test.txt"
        # X_test_path = "utilities/something/something.txt"
        # y_train_path = DATASET_PATH + "Y_train.txt"
        # y_test_path = DATASET_PATH + "Y_test.txt"
        # n_steps = 32 # 32 timesteps per series
        # n_steps = 1 # 32 timesteps per series
        # X_train = load_X(X_train_path)
        # X_test = activity_human.load_X(X_test_path)
        # X_test = activity_human.load_XLive(human_keypoint)
        #print X_test
        # y_train = load_y(y_train_path)
        # y_test = activity_human.load_y(y_test_path)
        # proof that it actually works for the skeptical: replace labelled classes with random classes to train on
        #for i in range(len(y_train)):
        #    y_train[i] = randint(0, 5)
        
        # Input Data 
        # n_input = len(X_train[0][0])  # num input parameters per timestep
        # training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
        # test_data_count = len(X_test)  # 1197 test series
        
        self.n_input = 36
        
        self.n_hidden = 34 # Hidden layer num of features
        # n_classes = 6
        n_classes = len(self.LABELS)
        # n_steps = 32
        
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
        self.x = tf.placeholder(tf.float32, [None, n_steps, self.n_input])
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
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.pred)) + l2 # Softmax loss
        
        if decaying_learning_rate:
            learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)
            
        #decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) #exponentially decayed learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step) # Adam Optimizer
        # correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(y,1))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # if decaying_learning_rate:
        #     learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)
        
        test_losses = []
        test_accuracies = []
        train_losses = []
        train_accuracies = []
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        # self.sess = tf.self.session(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        
        # training_iters = training_data_count *30
        #create saver before training
        saver = tf.train.Saver(var_list={'wh':weights['hidden'], 'wo':weights['out'], 'bh':biases['hidden'], 'bo':biases['out']})
        load = True
        train = False
        update = False
        #check if you want to retrain or import a saved model
        
        print("aaa")
        if load:
            saver.restore(self.sess, DATASET_PATH + "model.ckpt")
            print("Model restored.")
        print("bbb")
        
        correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
        
    # Load the networks inputs
    def runinference(self, human_keypoint):
        time_start = time.time()    
        ##### Inferencing
        # X_infer_path = "utilities/something/something.txt"
        # X_infer_path = DATASET_PATH + "X_test.txt"
        # X_val = load_X(X_infer_path)
        X_test = activity_human.load_XLive(human_keypoint)
        self.preds = self.sess.run(
            [self.pred],
            feed_dict={
                self.x: X_test
           }
        )
        
        id, self.conf = max(enumerate(self.preds[0][0]), key=operator.itemgetter(1))
        self.action = self.LABELS[id]
        
        print(self.preds, self.action)
        
        time_stop = time.time()
        print("TOTAL TIME:  {}".format(time_stop - time_start))
        
    def load_X(X_path):
        file = open(X_path, 'r')
        X_ = np.array(
            [elem for elem in [
                row.split(',') for row in file
            ]], 
            dtype=np.float32
        )
        file.close()
        blocks = int(len(X_) / n_steps)
        X_ = np.array(np.split(X_,blocks))
        return X_ 

    # Load the networks outputs
    def load_XLive(keypoints):
        # print(keypoints)
        
        print(len(keypoints), ":", [len(row) for row in keypoints])
        
        X_ = np.array(keypoints,dtype=np.float32)
        
        blocks = int(len(X_) / n_steps)
        X_ = np.array(np.split(X_,blocks))
        return X_ 

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

        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        _X = tf.reshape(_X, [-1, self.n_input])   
        # Rectifies Linear Unit activation function used
        _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, n_steps, 0) 

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
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
     
     
if __name__ == '__main__':
    mainhuman_activity()

