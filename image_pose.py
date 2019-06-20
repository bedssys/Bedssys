import argparse
import logging
import time
import json
import imutils

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

BLACK = [0, 0, 0]

# Square masking, to hide unwanted detection [(x0, y0), (x1, y1)]
DOMASK = True
MASK = [[(174, 0), (250, 80)],
        [(320, 0), (380, 50)],
        [(160, 140), (220, 206)],
        [(180, 155), (240, 230)]]

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
    parser.add_argument('--image', type=str, default='')
    parser.add_argument('--rotate', type=int, default=0) # Rotate CW
    parser.add_argument('--resize', type=str, default='576x288', help='network input resolution. default=432x368')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    parser.add_argument('--stats', type=bool, default=True, help='Display FPS, frame, etc.')
    parser.add_argument('--crop', type=int, default=-1, help='Crop a 2x2 collage image, -1 to disable.')
    args = parser.parse_args()

    # # Frame management
    # cap = cv2.VideoCapture()
    # tot_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # frame_skipped = 0
    # frame = 0
    i = 0
    
    # Model initiation
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
        
    # Reads image
    raw = cv2.imread(args.image)

    # if cap.isOpened() is False:
        # print("Error opening video stream or file")
    while True:
        # ret_val, raw = cap.read()
        # raw = imutils.rotate_bound(raw, args.rotate)
        
        h, w = raw.shape[:2]
        
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
            
        # Draw a mask around unwanted area
        if DOMASK and crop != -1:
            cv2.rectangle(image, MASK[crop][0], MASK[crop][1], BLACK, thickness=cv2.FILLED)

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=True)    # Copy the image rather than stacking one on top of another

        cv2.imshow('tf-pose-estimation result', image)
        cv2.imwrite( 'posedata/' +'.png',image)
        
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
