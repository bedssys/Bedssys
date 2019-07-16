import cv2
import numpy as np
import csv

LABELS = [
    "jalan_DR", "jalan_UR", "jalan_DL", "jalan_UL",
    "barang2_DR", "barang2_UR", "barang2_DL", "barang2_UL",
    "barang1l_DR", "barang1l_UR", "barang1l_DL", "barang1l_UL",
    "barang1r_DR", "barang1r_UR", "barang1r_DL", "barang1r_UL",
    "idle_ND"
]

# Similar to normalizeonce, but point-wise.
n_steps = 5

name = "_train.txt"

# The relative origin coord of each point
xo = np.zeros(18)
yo = np.zeros(18)

CENTER = True

# The blank image
W = 1024
H = 576

fy = open("Y" + name, "r").readlines()
fx = open("X" + name, "r").readlines()

xarr = []

# Line iterator
ly = 0
lx = 0

while True:
    canvas = np.zeros((H,W,3), np.uint8)

    # Remove newline, split by comma, conv to numpy array of type float
    skel = np.array( fx[lx][:-1].split(','), dtype="float64" )
    
    ly = int(lx/n_steps)
    lab = int(fy[ly][:-1])  # Get the label by reading floor(lx/n) of fy
    
    # As the multiplier, zero stays zero
    zero = [0 if k==0 else 1 for k in skel]
    
    xsk = skel[0::2]
    ysk = skel[1::2]

    # Set the coord origin to the center of image
    if CENTER:
        xsk += W/2
        ysk += H/2
    
    xsk *= zero[0::2]
    ysk *= zero[1::2]
    
    for (x, y) in zip(xsk, ysk):
        if not (x == 0 and y == 0):
            cv2.circle(canvas, (int(round(x)),int(round(y))), 3, (64,64,64), thickness=3, lineType=8, shift=0)
    
    vt = 20
    cv2.putText(canvas, "Y Line: %d %s" % (ly+1, LABELS[lab-1]),
        (10, vt),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    vt += 20
    cv2.putText(canvas, "X Line: %d" % (lx+1),
        (10, vt),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('Bedssys', canvas)
    k = cv2.waitKey(0)  # Wait for keypress infinitely
    print(k)    # Print keypress code
    if k == 27: # Esc, stop
        break
    elif k == ord('d'):
        lx += 1
    elif k == ord('a'):
        lx -= 1
    elif k == ord('D'):
        lx += n_steps
    elif k == ord('A'):
        lx -= n_steps
    
cv2.destroyAllWindows()

# Reset position

fx.close()
fy.close()