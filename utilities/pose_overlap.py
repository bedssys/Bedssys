dir = "overlap/"
input = "in.txt"
output = "out.txt"

# WARNING, make sure there's no header or unnecessary line to the file

npose = 5   # Pose per gesture
nover = 2   # Overlap

def pose_overlapper():
    # File handling
    in_file = dir + input
    out_file = dir + output
    
    fout = open(out_file, 'w').close() # Clear existing file
    fin = open(in_file, 'r')    # Open in read mode
    fout = open(out_file, 'w') # Open in append mode
    
    source = []
    
    # Read all the source data
    for x in fin:
      source.append(x)
    
    fin.close()
    
    if npose <= 2*nover:
        return
    
    #for n in range(len(source)):    
    # Modifying the iterator used for "FOR"
    # isn't legal and not working
    
    p = 1   # The actual counter, the first one is 1
    n = 0   # Source iterator
    while n < len(source):
        fout.write(source[n])
        # fout.write('\n')
        # Not necessary since the source is a text with newline
        
        # Overlap by stepping back the incrementation
        if p == npose:
            n -= nover
            p = 0
        
        p += 1
        n += 1
    
    fout.close()

if __name__ == '__main__':
    pose_overlapper()