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

scaler = joblib.load('scale_model_V2.pkl')
model = joblib.load('blink_model_V2.pkl')


#Intensity calculator using predefined and well known formula
def get_pixel_intensity(pixel_bgr):
    #unpack pixel_bgr
    b, g, r = pixel_bgr
    #the standard Luminance formula
    intensity = 0.299*r + 0.587*g + 0.114*b
    return intensity

def create_file(filename):
    
    with open(filename, mode = 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "EAR", "EAR Delta", "BlinkLabel", "Duration", "Min EAR", "Velocity", "Acceleration"])

def write_to_file(filename, TimeStamp, EAR_Val = 0 , change = 0, Blink_Label = 0, duration = -1, min_EAR = 1.0, velocity = 0, acceleration = 0):
    with open(filename, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([TimeStamp, EAR_Val, change, Blink_Label, duration, min_EAR, velocity, acceleration])

#Timer Starter
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

#Initialising Key Values
frames = 0
blinked_frames = 0
blink_count = 0
blink_start = 0
blink_end = 0
right_indexes = [362, 263, 385, 380, 387, 373]  
left_indexes = [33, 133, 160, 144, 158, 153] 
min_EAR_in_blink = 1.0
prev_EAR = 0
prev_velocity = 0
prev_frame_time = time.time()


#Begins recording
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.75, min_tracking_confidence = 0.75) as face_mesh:
    
    #The Light requirements - to be modified
    light_crit_low = 40
    light_low = 65
    light_high = 175
    light_crit_high = 210
    valid_dist_center = 0.20
    stable_time = -1
    stable_start = 2
    light_eligible = False
    stable = False

    while cap.isOpened():
        rec, frame = cap.read()
        
        if not rec:
            continue
        
        flipped_frame = cv2.flip(frame, 1)
        h, w, _ = flipped_frame.shape

        # Convert for MediaPipe
        rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
        flipped_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        flipped_frame.flags.writeable = True
        flipped_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        brightness = 0
        face_centered = False

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            #Face centering system
            nose = landmarks.landmark[0]
            face_x, face_y = nose.x, nose.y

            dist_from_center = math.dist((face_x,face_y),(0.5,0.5))
            
            #Eye lighting check system
            eye_intensities = []
            drawing_points = []
            for point_counter in right_indexes + left_indexes:
                #Extracting each index of the eyes
                point = landmarks.landmark[point_counter]
                #converting mediapipe float to pixel value
                cx_raw, cy_raw = int(point.x*w), int(point.y*h)
                #Ensuring pixel value is within bounds
                cx, cy = max(0, min(cx_raw, w - 1)), max(0, min(cy_raw, h - 1))
                #Draws a circle at the eye points
                cv2.circle(flipped_frame, (cx,cy), 2, (40,255,0), 1)
                drawing_points.append((point.x, point.y))
                #Extracts bgr list for the colours at the point using Matrix Notation
                colour_at_point = flipped_frame[cy, cx]

                eye_intensities.append(get_pixel_intensity(colour_at_point))

            eye_brightness = np.mean(eye_intensities)

            #Brightness bar
            bar_height = int((eye_brightness / 255) * 200)
            cv2.rectangle(flipped_frame, (w-50, h-50), (w-30, h-250), (50, 50, 50), -1)
            cv2.rectangle(flipped_frame, (w-50, h-50), (w-30, h-50-bar_height), (255, 255, 0), -1) 
            cv2.putText(flipped_frame, "BRIGHT", (w-80, h-260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if dist_from_center <= valid_dist_center:
                #Draw a rectangle showing the 40-60% zone
                start_point = (int(w * 0.3), int(h * 0.2))
                end_point = (int(w * 0.7), int(h * 0.8))
                cv2.rectangle(flipped_frame, start_point, end_point, (0, 255, 0), 1)
          
            #Determine the status
            if dist_from_center > valid_dist_center:
                #Draw a rectangle showing the 40-60% zone
                start_point = (int(w * 0.3), int(h * 0.2))
                end_point = (int(w * 0.7), int(h * 0.8))
                cv2.rectangle(flipped_frame, start_point, end_point, (0, 0, 255), 1)
                alert = "Face not centered, move face toward middle of the frame"
                color = (0, 0, 255) 
                light_eligible = False
            #Red Condition
            elif eye_brightness <= light_crit_low:
                alert = "CRITICAL: Too dark to detect eyes"
                color = (0, 0, 255) 
                light_eligible = False
            #Yellow condition
            elif light_crit_low < eye_brightness <= light_low:
                alert = "Slightly Low Light: Proceed with caution"
                color = (0, 255, 255)
                light_eligible = True
            #Green Condition
            elif light_low < eye_brightness < light_high:
                alert = "PERFECT CONDITIONS: Ready to test"
                color = (0, 255, 0)
                light_eligible = True

            #Orange Condition
            elif light_high < eye_brightness <= light_crit_high:
                alert = "Slightly High Light: May cause error in reflex test"
                color = (0, 165, 255)
                light_eligible = True

            #Red condition
            elif eye_brightness > light_crit_high: 
                alert = "CRITICAL: Light too bright (Washed out)"
                color = (0, 0, 255)
                light_eligible = False
            else:
                pass
            
            #Timer system for light and centralising stability
            if light_eligible:
                if stable_time == -1:
                    stable_time = time.time()
                elapsed = time.time() - stable_time

                if elapsed > stable_start:
                    alert = "STABLE! Press 'c' to start"
                    color = (0,255,0)
                    stable = True
                else:
                    alert = f"HOLD STILL for {round(stable_start - elapsed, 1)}s more"
                    stable = False
            else:
                stable_time = -1
                stable = False

            cv2.putText(flipped_frame, alert, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            #Display lightning check
            cv2.imshow("Lighting Optimisation", flipped_frame)
        else:
            cv2.putText(flipped_frame, "FACE NOT FOUND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Lighting Optimisation", flipped_frame)
            stable_time = -1

        #cv2 safety net
        key = cv2.waitKey(1) & 0xff

        if stable and key == ord('c'):
            break
        elif key == ord('x'):
            cap.release()
            cv2.destroyAllWindows()
            exit()


cv2.destroyWindow("Lighting Optimisation")

time.sleep(2)

#Stopwatch reset
start_time = time.time()

#Using FaceMesh class
with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.75, min_tracking_confidence = 0.75) as face_mesh:

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
                
                #Lists for drawing
                R_XYs = []
                L_XYs = []

                #Right Eye landmark coordinates
                for r in right_indexes:
                    point = face_landmark.landmark[r]
                    cx_raw, cy_raw = int(point.x * width), int(point.y * height)
                    cx, cy = max(0, min(cx_raw, width - 1)), max(0, min(cy_raw, height - 1))
                    cv2.circle(video, (cx,cy), 2, (40,255,0), 1)
                    R_XYs.append((point.x, point.y))
                
                #Left Eye landmark coordinates 
                for l in left_indexes:
                    point = face_landmark.landmark[l]
                    cx_raw, cy_raw = int(point.x * width), int(point.y * height)
                    cx, cy = max(0, min(cx_raw, width - 1)), max(0, min(cy_raw, height - 1))
                    cv2.circle(video, (cx,cy), 2, (40,255,0), 1)
                    L_XYs.append((point.x, point.y))

                #EAR Calculations
                R_EAR = EARratio(R_XYs)
                L_EAR = EARratio(L_XYs)
                avg_EAR = (R_EAR + L_EAR)/2


                #Time difference (dt)
                current_frame_time = time.time()
                dt = current_frame_time - prev_frame_time if (current_frame_time - prev_frame_time) > 0 else 0.03
                prev_frame_time = current_frame_time

                #Displacement (EAR Delta)
                del_EAR = avg_EAR - prev_EAR
                prev_EAR = avg_EAR # Update for next frame

                #Velocity & Acceleration
                if dt > 0.001:
                    current_velocity = del_EAR / dt
                    # Acceleration is Change in Velocity over Time
                    crnt_acceleration = (current_velocity - prev_velocity) / dt
                else:
                    current_velocity = 0
                    crnt_acceleration = 0

                prev_velocity = current_velocity # Update for next frame
                
                #Time stamp
                time_since_start = time.time() - start_time 

                #using ML format data
                clues = pd.DataFrame([[avg_EAR, del_EAR, current_velocity, crnt_acceleration]], columns=['EAR', 'EAR Delta', 'Velocity', 'Acceleration'])
                scld_clues = scaler.transform(clues)

                #Debugging method
                #cv2.putText(video, str(round(avg_EAR,3)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
                
                #[0] to extract only the label
                prediction = model.predict(scld_clues)[0]

                #I strategy of counting a blink when eyes go down(1)
                if prediction == 1:
                    if blinked_frames == 0:
                        blink_count += 1
                        blink_start = time.time()
                        cv2.putText(video, "BLINK!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    blinked_frames += 1
                    blink_dur = -1
                    #track the lowest EAR seen during this specific blink
                    if avg_EAR < min_EAR_in_blink:
                        min_EAR_in_blink = avg_EAR
                else:
                    if blinked_frames > 0:
                        blinked_frames = 0 
                        blink_end = time.time()
                        blink_dur = blink_end - blink_start
                    else:
                        blink_dur = -1
                        min_EAR_in_blink = 1.0
                
                #Fills CSV File
                write_to_file('blinks.csv', time_since_start, avg_EAR, del_EAR, prediction, blink_dur, min_EAR_in_blink, current_velocity, crnt_acceleration)

                #Blink count on screen for debugging
                cv2.putText(video, str(blink_count), (560, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)       
        
        #Temporary timer
        if time.time() - start_time > 150:
            break

        #Closes code when x is pressed
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

        #Displays video with mesh
        cv2.imshow("Blink Detection", video)

#Calculates end time
end_time = time.time()

#Neccesary to stop using webcam
cap.release()
    
#Closes webcam window
cv2.destroyAllWindows()

#Blink Summary
if blink_count > 0:
    dur_file = pd.read_csv('blinks.csv')
    working_data = dur_file[dur_file['Duration'] >= 0]['Duration']
    avg_blink = round(np.mean(working_data),5)
    longest_blink = round(np.max(working_data),5)
    shortest_blink = round(np.min(working_data),5)
    
    #Grouping all the blink events
    blink_groups = dur_file[dur_file['BlinkLabel'] == 1].groupby('Timestamp')

    #Find the fastest closing velocity for each event
    individual_blink_speeds = blink_groups['Velocity'].min()

    #Average for average peak velocity
    avg_peak_velocity = individual_blink_speeds.mean()
    
    #Rate Calculator
    seconds = end_time - start_time
    blink_rate = round(blink_count/seconds, 5)

    print(f"Average Blink Rate: {blink_rate*60} blinks/min")
    print(f"Average Blink Duration: {avg_blink*1000} ms")
    print(f"Longest Blink: {longest_blink*1000} ms")
    print(f"Shortest Blink: {shortest_blink*1000} ms")
    print(f"Average Peak Velocity: {avg_peak_velocity} units/sec")
else:
    print("No blinks were detected")
