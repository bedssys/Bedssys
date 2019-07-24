import numpy as np
import csv
import random

# Total augmentation rather than the traditional full-pose gesture simple augmentation.
# Make sure to have the same gestures repeated/simple-augmented already in the file.

n_steps = 5

# Amount of copy created in certain schemes
pt_ori = 0
pt_rand = 12
pt_rem = 0

# Param
randval = 1024./100
remval = 5

dir = "augment/"
names = ["_train.txt", "_test.txt"]

for name in names:
    print('\n',name,'\n')
    fy = open("Y" + name, "r")
    fx = open("X" + name, "r")
    fyo = open(dir + "Y" + name, "w")
    fxo = open(dir + "X" + name, "w")

    xarr = []
    yarr = []
    
    # Copy
    ln = 0
    fxc = []
    
    for x in fx:
        fxc.append(x)

    for line in fy:
        ynum = int(line[:-1])
        
        if ynum != 17:
            ori = pt_ori
            rand = pt_rand
            rem = pt_rem
        else: # Idle gesture
            ori = 1
            rand = 0
            rem = pt_rem
            
        mode = 0  # Original, rand, rem
        total = ori + rand + rem
        
        for i in range(total):
            yarr.append(line)
            
            if ori == 0:
                mode = 1
                ori = -1
            # elif rand == 0:
                # mode = 2
                # rand = -1
            # elif rem == 0:
                # mode = 3
                # rem = -1
            
            for i in range(n_steps):
                # Remove newline, split by comma, conv to numpy array of type float
                skel = np.array( fxc[ln][:-1].split(','), dtype="float64" ) 
                ln += 1
                
                if mode == 0:
                    # Duplicate the original to output
                    xarr.append(','.join(map(str, skel)) + "\n")
                else:
                    x = skel[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]]
                    y = skel[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]]
                    
                    nzero_x = sum(1 if (k != 0) else 0 for k in x)
                    nzero_y = sum(1 if (k != 0) else 0 for k in y)
                    
                    if (nzero_x == 0):
                        nzero_x = 1
                    if (nzero_y == 0):
                        nzero_y = 1
                    
                    # Augmentation
                    if mode == 1:
                        # Randomize location, float values
                        x += np.random.rand(18) * 2*randval - randval
                        y += np.random.rand(18) * 2*randval - randval
                        rand -= 1
                    elif mode == 2:
                        # Remove rand points
                        print("not yet")
                        rem -= 1
                    
                    # As the multiplier, zero stays zero
                    zero = [0 if x==0 else 1 for x in skel]
                    
                    # Recombine, placed one after another
                    skel[0::2] = x
                    skel[1::2] = y
                    
                    skel = skel * zero        
                    xarr.append(','.join(map(str, skel)) + "\n")
            
            ln -= n_steps
            
            if mode == 0:
                ori -= 1
            elif mode == 1:
                rand -= 1
        
        ln += n_steps   # To nullify the previous decrement if done

    # Print to file
    for line in xarr:
        fxo.write(line)
    for line in yarr:
        fyo.write(line)
        
    fx.close()
    fy.close()
    fxo.close()
    fyo.close()