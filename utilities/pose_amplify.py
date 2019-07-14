import numpy as np
import csv

n_steps = 5

names = ["_train.txt", "_test.txt"]

for name in names:
    fx = open("X" + name, "r+")
    fy = open("Y" + name, "r+")

    xarr = []
    yarr = []

    xtot = 0
    ytot = 0

    # Set => list with unique values & ordered.
    # So, way faster if used for searching.
    NE = set([1,5,9,13])
    NW = set([2,6,10,14])
    SE = set([3,7,11,15])
    SW = set([4,8,12,16])

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
            
            zero = [0 if x==0 else 1 for x in skel] # As the multiplier, zero stays zero
            
            x = skel[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]] + sx
            y = skel[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]] + sy
            
            # Recombine, placed one after another
            skel[0::2] = x
            skel[1::2] = y
            
            skel = skel * zero
            xarr.append(','.join(map(str, skel)) + "\n")
        
        yarr.append(line)

    # Reset position
    fx.seek(0)
    fy.seek(0)

    # Print back to file
    for line in xarr:
        fx.write(line)
    for line in yarr:
        fy.write(line)

    # Remove the remaining text from previous file
    fx.truncate()
    fy.truncate()

    fx.close()
    fy.close()

    # print(y)
    # print(x)