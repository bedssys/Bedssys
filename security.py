class Frame:
    # Update this class for every components used in the calculation
    __slots__ = ('act_labs', 'act_confs', 'dobj', 'face_names', 'level')
    
    # Recognized gestures counted as negative
    NEGLABS = {"curiga_DR", "curiga_UR", "curiga_DL", "curiga_UL"}
    NEGOBJ = {"something"}

    def __init__(self, act_labs, act_confs, dobj, face_names):
        self.act_labs = act_labs
        self.act_confs = act_confs
        self.dobj = dobj
        self.face_names = face_names
        self.level = -1     # Uncalculated value
        
    def calc(self):
        # The calculation depth is basically endless.
        # Some ideas to consider:
        # - Pairing obj location with person/pose
        # - Check face detection with detected person/pose
        # - Positive behaviors
        
        all = 0    # Count of total componenets
        neg = 0    # Count of negative componenets
        
        for (lab, conf) in zip(self.act_labs, self.act_confs):
            if lab in self.NEGLABS:
                neg += 1
            elif conf < 0.3:
                neg += 1
            all += 1
        
        for obj in self.dobj:
            # An array: obj = [label, confidence, bounds]
            if obj[0] in self.NEGOBJ:
                neg += 1
            all += 1
            
        for face in self.face_names:
            if face == "Unknown":
                neg += 1
            all += 1
        
        # Get security conclusion of a single frame
        self.level = (all-neg)/all
        
        
    
    
    
    # def __init__(self, pairs):
        # self.pairs = []
        # self.uidx_list = set()
        # self.body_parts = {}
        # for pair in pairs:
            # self.add_pair(pair)
        # self.score = 0.0