import numpy as np
import csv

names = ["_train.txt", "_test.txt"]

for name in names:
    fx = open("X" + name, "r+")

    xarr = []

    for line in fx:
        # Remove newline, split by comma, conv to numpy array of type float
        skel = np.array( line[:-1].split(','), dtype="float64" )
        xarr.append(','.join([("%.6f" % x) for x in skel]) + "\n")

    # Reset position
    fx.seek(0)

    # Print back to file
    for line in xarr:
        fx.write(line)

    # Remove the remaining text from previous file
    fx.truncate()
    fx.close()

    # print(y)
    # print(x)