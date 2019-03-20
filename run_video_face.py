# Use this one as a template for processing a video & output images.
# If you're using the images only for checking (not to be used by a computer system),
# always use JPG format with some compression instead of PNG.

import argparse
import logging
import time
import json
import imutils
# import os

import cv2
import numpy as np

import facerec.recognize as fr

fps_time = 0
# cd = os.path.dirname(os.path.realpath(__file__))

BLACK = [255, 255, 255]

# For frame skipping
REAL_FPS = 30
PROC_FPS = 1
SKIP_FRAME = round(int(REAL_FPS/PROC_FPS)) - 1

# WARNING, make sure the folder exist
# output = "X_face.txt"
output_dir = "./test_face/rian/"

def round_int(val):
    return (round(val, 3))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--rotate', type=int, default=0) # Rotate CW
    args = parser.parse_args()
    
    frame_skipped = 0
    frame = 0

    cap = cv2.VideoCapture(args.video)
    tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret_val, image = cap.read()
    
    h, w, c = image.shape
    print("WxH:", w, h)
    
    # Saving texts
    # open(output, 'w').close() # Clear existing file
    # fp = open(output, 'a+') # Open in append mode
    
    facer = fr.face_recog(face_dir="./facerec/face/")

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, image = cap.read()
        image = imutils.rotate_bound(image, args.rotate)
        
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)           # Reduce initial file size to match real-time webcam
        
        # Skip frames to get realtime data representation
        if frame_skipped < SKIP_FRAME:
            frame += 1
            frame_skipped += 1
            continue
        
        frame += 1
        frame_skipped = 0

        # Face recognition
        face_locs, face_names = facer.runinference(image, tolerance=0.6, prescale=0.5, upsample=2)
        print(face_locs, face_names)
        
        # GUI
        vt = 20  # Vertical position
        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, vt),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        vt += 20
        cv2.putText(image, "Frame: %d/%d" % (frame, tot_frame), (10, vt),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        image = fr.face_recog.display(image, face_locs, face_names, prescale=0.5)
        
        tout = output_dir + ("%05d" % frame) +'.jpg'
        print(tout)
        
        # Save images
        # image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)           # Reduce final file size
        # cv2.imwrite(tout, image)                                  # Output full-quality image
        cv2.imwrite(tout, image, [cv2.IMWRITE_JPEG_QUALITY, 90])    # Output compressed image
        cv2.imshow('run video result', image)
        fps_time = time.time()
        
        # Save texts
        # str_q = str(item)[1 : -1]
        # print(str_q)
        # fp.write(str_q)
        # fp.write('\n')
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    # fp.close()
    