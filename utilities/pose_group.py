import re

n_steps = 5

# Validations
md_lim = 4          # Minimum count of the most label in a group to be valid
trash = {0, 9999}   # Discarded labels

# Format is tsv (tab separated)
names = ["train.txt", "test.txt"]

LABELS = [
    "jalan_DR", "jalan_UR", "jalan_DL", "jalan_UL",
    "barang2_DR", "barang2_UR", "barang2_DL", "barang2_UL",
    "barang1l_DR", "barang1l_UR", "barang1l_DL", "barang1l_UL",
    "barang1r_DR", "barang1r_UR", "barang1r_DL", "barang1r_UL",
    "idle_ND"
]

for name in names:
    fin = open(name, "r")
    fx = open("X_" + name, "w")
    fy = open("Y_" + name, "w")
    
    fx.seek(0)
    fy.seek(0)

    x = []
    y = []
    
    # Gestures
    ls = []
    ps = []
    n = 0

    # Stats
    ltot = [0] * len(LABELS) # Zeros array of length LABELS

    for line in fin:
        # Get the components
        # Split & strip the \t (tab)
        # Newline is also split here
        comp = re.split(r'\t+', line.rstrip('\t'))
        
        # Number, Label, Pose
        lab = int(comp[1])
        pos = comp[2]
        
        ls.append(lab)
        ps.append(pos)
        
        n += 1
        
        # Group obtained
        if n == n_steps:
            mode = max(set(ls), key=ls.count)
            
            # Check if there's no pose with character "_" only
            # Check if the mode is fulfilling the minimum limit
            # Check if the mode is not in trash labels
            if (sum([1 if (pose[0] == "_") else 0 for pose in ps]) < 1 and
                sum([1 if (x == mode) else 0 for x in ls]) >= md_lim and
                mode not in trash):
                
                # Write to file directly
                fy.write(str(mode) + "\n")
                for pose in ps:
                    if (pose[-1] == "\n"):
                        fx.write(pose)
                    else:
                        fx.write(pose + "\n")
                    
                ltot[mode-1] += 1
            
            # Reset
            ls = []
            ps = []
            n = 0

    print(ltot, sum(ltot))
    
    fin.close()
    fx.close()
    fy.close()