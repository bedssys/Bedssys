import numpy as np
import csv

n_steps = 5

names = ["_train.txt", "_test.txt"]

for name in names:
    fx = open("X" + name, "r")
    fy = open("Y" + name, "r")

    xtot = 0
    ytot = 0

    # Set => list with unique values & ordered.
    # So, way faster if used for searching.
    NE = set([1,5,9,13])
    NW = set([2,6,10,14])
    SE = set([3,7,11,15])
    SW = set([4,8,12,16])

    # SUBIM = [512,288]           # Sub-image dimension
    SUBIM = [512+999,288+999]   # Sub-image dimension + amplification

    valid = 0
    invalid = 0

    xline = 1

    for line in fy:
        lab = int(line[:-1]) # Read w/o the last char (newline)
        
        # Determine the shift based on the labels
        # In real implementation, this would be done
        # by checking the pose center value
        if lab in NE:
            sx = 0
            sy = 0
        elif lab in NW:
            sx = 1000
            sy = 0
        elif lab in SE:
            sx = 0
            sy = 1000
        elif lab in SW:
            sx = 1000
            sy = 1000
        else:
            sx = 0
            sy = 0

        for i in range(n_steps):
            # Remove newline, split by comma, conv to numpy array of type float
            skel = np.array( fx.readline()[:-1].split(','), dtype="float64" )
            
            # Calculate the midpoint representation, using average    
            # (Exact copy from average function)
            # Remember that a point might not be detected, giving zero. Count the non-zero.
            # Below line is equivalent to COUNTIF(not-zero). Lazy: Buggy if there's an actual poin on axis.
            
            x = skel[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]]
            y = skel[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]]
            
            nzero_x = sum(1 if (k != 0) else 0 for k in x)
            nzero_y = sum(1 if (k != 0) else 0 for k in y)
            
            if (nzero_x != 0):
                if (nzero_y == 0):
                    nzero_y = 1
                
                ax = sum(x) / nzero_x
                ay = sum(y) / nzero_y
                
                # Check
                if (ax <= SUBIM[0]) and (ay <= SUBIM[1]):
                    if lab in NE:
                        valid += 1
                    else:
                        invalid += 1
                        print(lab, xline, "%.2f %.2f" % (ax, ay))
                elif (ax > SUBIM[0]) and (ay <= SUBIM[1]):
                    if lab in NW:
                        valid += 1
                    else:
                        invalid += 1
                        print(lab, xline, "%.2f %.2f" % (ax, ay))
                elif (ax <= SUBIM[0]) and (ay > SUBIM[1]):
                    if lab in SE:
                        valid += 1
                    else:
                        invalid += 1
                        print(lab, xline, "%.2f %.2f" % (ax, ay))
                elif (ax > SUBIM[0]) and (ay > SUBIM[1]):
                    if lab in SW:
                        valid += 1
                    else:
                        invalid += 1
                        print(lab, xline, "%.2f %.2f" % (ax, ay))
                else:
                    invalid += 1
                    print(lab, xline, "%.2f %.2f" % (ax, ay))
        
            xline += 1
        
    print(valid, invalid)
        
    fx.close()
    fy.close()

    # print(y)
    # print(x)