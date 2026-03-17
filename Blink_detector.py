#Python File
#Imports
import cv2
import mediapipe as mp
import math
import time
import csv
import os
import pandas as pd
import numpy as np
import joblib

scaler = joblib.load('scale_model.pkl')
model = joblib.load('blink_model.pkl')



def create_file(filename):
    
    with open(filename, mode = 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "EAR", "EAR Delta", "BlinkLabel", "Duration"])

def write_to_file(filename, EAR_Val = 0 , change = 0, Blink_Label = 0, duration = -1):
    with open(filename, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([time.time(), EAR_Val, change, Blink_Label, duration])


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


opt = cv2.VideoCapture(0)
#Capturing Video


label = 0

#Drawing style
style1 = mp_drawing.DrawingSpec(color=(0,0,255), thickness = 1, circle_radius = 1)
style2 = mp_drawing.DrawingSpec(color=(25,255,0), thickness = 1, circle_radius = 1)


#Initialising Key Values
frames = 0
blinked_frames = 0
blink_count = 0
blink_start = 0
blink_end = 0
right_indexes = [362, 263, 385, 380, 387, 373]  
left_indexes = [33, 133, 160, 144, 158, 153] 
'''
with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.75, min_tracking_confidence = 0.75) as face_mesh:
    create_file('calibrator.csv')
    while opt.isOpened():
        stat, module = opt.read()
        cal_time = time.time()
        if (cal_time-start_time) > 10:
            break
        if not stat:
            continue

        flipped_opt = cv2.flip(module, 1)
        rgb_opt = cv2.cvtColor(flipped_opt, cv2.COLOR_BGR2RGB)
        rgb_opt.flags.writeable = False
        CalRecData = face_mesh.process(rgb_opt)
        rgb_opt.flags.writeable = True
        flipped_opt = cv2.cvtColor(rgb_opt, cv2.COLOR_RGB2BGR)

        height, width, _ = flipped_opt.shape

        if CalRecData.multi_face_landmarks:
            for face_landmark in CalRecData.multi_face_landmarks:
                R_XYs = []
                L_XYs = []

                #Right Eye landmark coordinates
                for r in right_indexes:
                    point = face_landmark.landmark[r]
                    cx, cy = int(point.x * width), int(point.y * height)
                    cv2.circle(flipped_opt, (cx,cy), 2, (40,255,0), 1)
                    R_XYs.append((point.x, point.y))
                
                #Left Eye landmark coordinates 
                for l in left_indexes:
                    point = face_landmark.landmark[l]
                    cx, cy = int(point.x * width), int(point.y * height)
                    cv2.circle(flipped_opt, (cx,cy), 2, (40,255,0), 1)
                    L_XYs.append((point.x, point.y))

                #EAR Calculations
                R_EAR = EARratio(R_XYs)
                L_EAR = EARratio(L_XYs)
                avg_EAR = (R_EAR + L_EAR)/2

                write_to_file('calibrator.csv', avg_EAR)

                #Debugging method
                cv2.putText(flipped_opt, str(round(avg_EAR,3)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        
        cv2.imshow("Calibration", flipped_opt)
        if cv2.waitKey(1) & 0xff == ord('c'):
            break
cv2.destroyAllWindows()
opt.release()


# Math Block
data = pd.read_csv('calibrator.csv')

#Get only top 70% of data
open_data = data[data['EAR'] >= data['EAR'].quantile(0.3)]['EAR']

mean = np.mean(open_data)
std_dev = np.std(open_data)
close_thres = round((mean*0.37),3)
open_thres = round((mean*0.74),3)
'''
prev_EAR = 0

start_time = time.time()

cap = cv2.VideoCapture(0)
#Using FaceMesh class
with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.75, min_tracking_confidence = 0.75) as face_mesh:

    #blink_start = 0
    create_file('blinks.csv')
    #Video Loop
    while cap.isOpened():

        
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
                del_EAR = avg_EAR - prev_EAR
                prev_EAR = avg_EAR

                #using ML format data
                clues = pd.DataFrame([[avg_EAR, del_EAR]], columns=['EAR', 'EAR_Change'])
                scld_clues = scaler.transform(clues)

                #Debugging method
                cv2.putText(video, str(round(avg_EAR,3)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                
                
                
                #[0] to extract only the label
                prediction = model.predict(scld_clues)[0]


                #Alt strategy of counting a blink when eyes go down and not when they come back up 
                if prediction == 1:
                    if blinked_frames == 0:
                        blink_count += 1
                        blink_start = time.time()
                        cv2.putText(video, "BLINK!", (480, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    blinked_frames += 1
                    blink_dur = -1
                else:
                    if blinked_frames > 0:
                        blinked_frames = 0 
                        label = 0
                        blink_end = time.time()
                        blink_dur = round(blink_end - blink_start, 3)
                    else:
                        blink_dur = -1

                write_to_file('blinks.csv', avg_EAR, del_EAR, prediction, blink_dur)
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

dur_file = pd.read_csv('blinks.csv')
working_data = dur_file[dur_file['Duration'] >= 0]['Duration']
avg_blink = round(np.mean(working_data),3)
longest_blink = np.max(working_data)
shortest_blink = np.min(working_data)

#Speed Calculator
seconds = end_time - start_time

blink_speed = round(blink_count/seconds, 2)
print(blink_speed, "blinks/sec")
print(avg_blink, "s")
print(longest_blink, "s")
print(shortest_blink, "s")
#print(open_thres)
#print(close_thres)