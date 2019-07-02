import numpy as np
import csv

# Normalize only using the first pose of a gesture as origin.
n_steps = 8

names = ["_train.txt", "_test.txt"]

for name in names:
    fy = open("Y" + name, "r")
    fx = open("X" + name, "r+")

    xarr = []

    for line in fy:
        for i in range(n_steps):
            # Remove newline, split by comma, conv to numpy array of type float
            skel = np.array( fx.readline()[:-1].split(','), dtype="float64" )
            
            x = skel[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]]
            y = skel[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]]
            
            # Only calculate the mid-point for the first pose of a gesture
            if (i == 0):
                nzero_x = sum(1 if (k != 0) else 0 for k in x)
                nzero_y = sum(1 if (k != 0) else 0 for k in y)
                
                if (nzero_x == 0):
                    nzero_x = 1
                if (nzero_y == 0):
                    nzero_y = 1
                
                ax = sum(x) / nzero_x
                ay = sum(y) / nzero_y
        
            # Normalize, poses are now relative to the first pose (which is in origin)
            x -= ax
            y -= ay
            
            # As the multiplier, zero stays zero
            zero = [0 if k==0 else 1 for k in skel]
            
            # Recombine, placed one after another
            skel[0::2] = x
            skel[1::2] = y
            
            skel = skel * zero
            xarr.append(','.join(map(str, skel)) + "\n")

    # Reset position
    fx.seek(0)

    # Print back to file
    for line in xarr:
        fx.write(line)

    # Remove the remaining text from previous file
    fx.truncate()
    fx.close()

    fy.close()

    # print(y)
    # print(x)