n_steps = 8

remove = 13

names = ["_train.txt", "_test.txt"]

for name in names:
    fx = open("X" + name, "r+")
    fy = open("Y" + name, "r+")

    x = []
    y = []

    xtot = 0
    ytot = 0

    for line in fy:
        # print(line)
        
        # Other than removed data
        # Remember line has endline, 1 character
        num = int(line[:-1])
        
        if num != remove:
            # Decrement the number if bigger
            if (num > remove):
                num -= 1
            
            # Back to string & append to list
            y.append(str(num) + "\n")
            
            # For every action, there're n_steps keypoints
            for i in range(n_steps):
                x.append(fx.readline())
                
        else:
            # Read the text n_step times to skip
            ytot = ytot + 1
            for i in range(n_steps):
                xtot = xtot + 1
                next(fx)
        
    print(ytot)
    print(xtot)

    # Reset position
    fx.seek(0)
    fy.seek(0)

    # Print back to file
    for line in x:
        fx.write(line)
    for line in y:
        fy.write(line)

    # Remove the remaining text from previous file
    fx.truncate()
    fy.truncate()

    # print(y)
    # print(x)