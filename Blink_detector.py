#Imports
import cv2
import mediapipe as mp
import math
import time
import csv
import os

start_time = time.time()
#EAR Calculator 
def EARratio(coords):
    w = math.dist(coords[0],coords[1])
    h1 = math.dist(coords[2],coords[3])
    h2 = math.dist(coords[4],coords[5])
    EAR = (h1 + h2)/(w*2)
    return EAR

#Two Key functions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

#Capturing Video
cap = cv2.VideoCapture(0)



#Drawing style
style1 = mp_drawing.DrawingSpec(color=(0,0,255), thickness = 1, circle_radius = 1)
style2 = mp_drawing.DrawingSpec(color=(25,255,0), thickness = 1, circle_radius = 1)


#Initialising Key Values
frames = 0
blinked_frames = 0
blink_count = 0
right_indexes = [362, 263, 385, 380, 387, 373]  
left_indexes = [33, 133, 160, 144, 158, 153] 




#Using FaceMesh class
with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.75, min_tracking_confidence = 0.75) as face_mesh:

    
    
    #Video Loop
    while cap.isOpened():

        frames += 1
        #Read video data
        rec, frame = cap.read()
        
        #Avoiding webcam recording inconsistency
        if not rec:
            continue

        #Flipping to ensure mirror-like display
        flipped_vid = cv2.flip(frame, 1)

        #Converting BGR to RGB because mediapipe uses RGB but cv2 uses BGR
        video = cv2.cvtColor(flipped_vid, cv2.COLOR_BGR2RGB)
        
        #Making the colour swapped image unwriteable to make program more efficient and faster to read
        video.flags.writeable = False
        result = face_mesh.process(video)
        
        #Converting colour code format back to cv2 format (BGR)
        video.flags.writeable = True
        video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)

        #Calculates dimensions
        height, width, _ = video.shape
        

        
        if time.time() - start_time > 60:
            break
        
        
        #Condition just to avoid tracking errors
        if result.multi_face_landmarks:

            #Loop to draw a mesh and for values
            for face_landmark in result.multi_face_landmarks:
                #mp_drawing.draw_landmarks(image=video, landmark_list = face_landmark, connections=mp_face_mesh.FACEMESH_CONTOURS, connection_drawing_spec = style2, landmark_drawing_spec = style1)
                R_XYs = []
                L_XYs = []

                #Right Eye landmark coordinates
                for r in right_indexes:
                    point = face_landmark.landmark[r]
                    cx, cy = int(point.x * width), int(point.y * height)
                    cv2.circle(video, (cx,cy), 2, (255,0,0), 1)
                    R_XYs.append((point.x, point.y))
                
                #Left Eye landmark coordinates 
                for l in left_indexes:
                    point = face_landmark.landmark[l]
                    cx, cy = int(point.x * width), int(point.y * height)
                    cv2.circle(video, (cx,cy), 2, (255,0,0), 1)
                    L_XYs.append((point.x, point.y))

                #EAR Calculations
                R_EAR = EARratio(R_XYs)
                L_EAR = EARratio(L_XYs)
                avg_EAR = (R_EAR + L_EAR)/2

                #Debugging method
                cv2.putText(video, str(round(avg_EAR,3)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                
                #Code Block to count a blink
                if avg_EAR < 0.21:
                    blinked_frames += 1
                elif avg_EAR > 0.25:
                    if blinked_frames > 1:
                        blink_count += 1

                        #Blink 
                        cv2.putText(video, "BLINK!", (480, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) 
                    blinked_frames = 0
                    #blink = True
                    #break
                else:
                    pass
                    

                #Blink count on screen for debugging
                cv2.putText(video, str(blink_count), (560, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)       
        #if blink == True:
         #   break
        #Closes code when x is pressed
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

        #Displays video with mesh
        cv2.imshow("Face Tracking", video)

#Neccesary to stop using webcam
cap.release()
    
#Closes webcam window
cv2.destroyAllWindows()

end_time = time.time()
#Speed Calculator
seconds = end_time - start_time

blink_speed = round(blink_count/seconds, 2)
print(blink_speed, "blinks/sec")

