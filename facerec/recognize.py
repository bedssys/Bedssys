import face_recognition as fr
import time
import cv2
import os

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# # Load a sample picture and learn how to recognize it.
# obama_image = fr.load_image_file("./face/Dafa.jpg")
# obama_face_encoding = fr.face_encodings(obama_image)[0]

# # Load a second sample picture and learn how to recognize it.
# biden_image = fr.load_image_file("./face/William.jpg")
# biden_face_encoding = fr.face_encodings(biden_image)[0]

# # Create arrays of known face encodings and their names
# known_face_encodings = [
    # obama_face_encoding,
    # biden_face_encoding
# ]
# known_face_names = [
    # "Barack Obama",
    # "Joe Biden"
# ]

class face_recog:
    def __init__(self, face_dir="./facerec/face/"):
        # Loading all data in face folder
        files = os.listdir(face_dir)

        self.all_encs = []
        self.all_names = []

        for name in files:
            print(name)
            img = fr.load_image_file(face_dir + name)
            name = name[0 : -4]
            self.all_names.append(name)
            self.all_encs.append(fr.face_encodings(img)[0])
            
        # Initialize some variables
        self.process_this_frame = True
        
        
    def runinference(self, frame, tolerance=0.6, prescale=0.5, upsample=2):
        # Resize frame of video to 1/4 size for faster face recognition processing
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = cv2.resize(frame, (0, 0), fx=prescale, fy=prescale)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        # if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        # locs = fr.face_locations(rgb_small_frame)
        locs = fr.face_locations(rgb_small_frame, model="cnn", number_of_times_to_upsample=upsample)
        face_encodings = fr.face_encodings(rgb_small_frame, locs)

        names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces(self.all_encs, face_encoding, tolerance=tolerance)
            name = "Unknown"

            # If a match was found in all_encs, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = self.all_names[first_match_index]

            names.append(name)
        
        return(locs, names)
        # process_this_frame = not process_this_frame

    def display(frame, locs, names, prescale):
        # Display the results
        for (top, right, bottom, left), name in zip(locs, names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= int(1/prescale)
            right *= int(1/prescale)
            bottom *= int(1/prescale)
            left *= int(1/prescale)

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        return frame
        # cv2.imshow('Video', frame)
    
# if __name__ == "__main__":
    # facer = face_recog()
    
    # print(facer.runinference())
