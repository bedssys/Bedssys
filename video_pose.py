import argparse
import logging
import time
import json
import imutils

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

BLACK = [0, 0, 0]

# For frame skipping
REAL_FPS = 30
PROC_FPS = 30
SKIP_FRAME = round(REAL_FPS/PROC_FPS) - 1

DOMASK = True  
# Full 2x2 mode, masking areas to NOT be detected by openpose.
# Used to hide noisy area unpassable by human. (Masks are not shown during preview)
# The mask is a polygon, specify the vertices location.
PMASK = [   np.array([[610,520],[770,430],[960,576],[660,576]], np.int32),       # SW
            np.array([[185,430],[255,470],[70,570],[0,575],[0,530],[120,410]], np.int32),  # SE
            np.array([[760,200],[880,288],[1024,134],[985,44]], np.int32),       # NW
            np.array([[260,190],[50,50],[136,53],[327,157]], np.int32)           # NE
            ]  

# Cropping 2x2 video, -1 to disable
# CROP = 3

out_dir = "data/raw/"
output = "X_raw.txt"

def round_int(val):
    return (round(val, 3))

def write_coco_json(human, image_w, image_h):
    keypoints = []
    coco_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    for coco_id in coco_ids:
        if coco_id not in human.body_parts.keys():
            keypoints.extend([0, 0])
            continue
        body_part = human.body_parts[coco_id]
        keypoints.extend([round_int(body_part.x * image_w), round_int(body_part.y * image_h)])
    return keypoints
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--rotate', type=int, default=0) # Rotate CW
    parser.add_argument('--resize', type=str, default='512x288', help='network input resolution. default=432x368')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    
    parser.add_argument('--model', type=str, default='mobilenet_v2_small', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--stats', type=bool, default=True, help='Display FPS, frame, etc.')
    parser.add_argument('--crop', type=int, default=-1, help='Crop a 2x2 collage image, -1 to disable.')
    args = parser.parse_args()

    # Frame management
    cap = cv2.VideoCapture(args.video)
    tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_skipped = 0
    frame = 0
    i = 308
    
    # Model initiation
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    
    # File handling
    crop = args.crop
    out_file = out_dir + str(crop) + output
    
    open(out_file, 'w').close() # Clear existing file
    fp = open(out_file, 'a+') # Open in append mode
    
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    while cap.isOpened():
        ret_val, raw = cap.read()
        raw = imutils.rotate_bound(raw, args.rotate)
        
        h, w = raw.shape[:2]
            
        # Draw a polygon mask around unwanted area, for 4 cam mode
        if DOMASK:
            for pmask in PMASK:
                cv2.fillPoly(raw, [pmask], color=(0,0,0))
                
        # Cropping
        if crop == -1:
            image = raw
        elif crop == 0:
            image = raw[0:int(h/2), 0:int(w/2)] # Top-left
        elif crop == 1:
            image = raw[0:int(h/2), int(w/2):w] # Top-right
        elif crop == 2:
            image = raw[int(h/2):h, 0:int(w/2)] # Bot-left
        elif crop == 3:
            image = raw[int(h/2):h, int(w/2):w] # Bot-right
        
        # Skip frames to get realtime data representation
        if frame_skipped < SKIP_FRAME:
            frame += 1
            frame_skipped += 1
            continue
        
        frame += 1
        frame_skipped = 0
        
        # image = cv2.copyMakeBorder(image , 0, 0, 256, 256, cv2.BORDER_CONSTANT, value=BLACK)
        # image = cv2.copyMakeBorder(image_src , 0, 0, 256, 256, cv2.BORDER_REFLECT)

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        
        if not args.showBG:
            image = np.zeros(image.shape)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        if args.stats:
            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(image, "Frame: %d/%d" % (frame, tot_frame), (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('tf-pose-estimation result', image)
        cv2.imwrite( 'posedata/' + str(i) +'.png',image)
        
        i = i + 1
        fps_time = time.time()

        # Printing json
        image_h, image_w = image.shape[:2]
        count = 0
        item = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for human in humans:
            if (count == 0):
                item = write_coco_json(human,image_w,image_h)
            count = count + 1
        # json.dump(result, fp)
        # json.dump(result, fp)# slice off first and last character
        str_q = str(item)[1 : -1]
        # print(str_q)
        fp.write(str_q)
        fp.write('\n')
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    fp.close()
    
logger.debug('finished+')
