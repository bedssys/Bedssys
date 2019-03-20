n_steps = 5

remove = "3"

name = "_train.txt"
fx = open("X" + name, "r+")
fy = open("Y" + name, "r+")

x = []
y = []

xtot = 0
ytot = 0

for line in fy:
    # print(line)
    
    # Other than illegal data
    if line[0] != remove:
        # Decrement the number if bigger
        line = list(line)
        if(ord(line[0]) > ord(remove)):
            line[0] = chr(ord(line[0]) - 1)
        
        # Back to string & append to a list
        line = ''.join(line)
        y.append(line)
        
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