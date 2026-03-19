import cv2
import mediapipe as mp
import math
import time
import csv
import os
import pandas as pd
import numpy as np
import joblib



#Intensity calculator using predefined and well known ITU-R 601 formula
def get_pixel_intensity(pixel_bgr):
    #unpack pixel_bgr
    b, g, r = pixel_bgr
    #the standard Luminance formula
    intensity = 0.299*r + 0.587*g + 0.114*b
    return intensity

def create_file(filename):
    
    with open(filename, mode = 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(["Time Stamp", "Pupil Intensity"])

def write_to_file(filename, TimeStamp, pupil_intensity = 0):
    with open(filename, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([TimeStamp, pupil_intensity])

#Timer Starter
start_time = time.time()

#Two Key functions
mp_face_mesh = mp.solutions.face_mesh

#Initialising Key Values
right_indexes = [362, 263, 385, 380, 387, 373]  
left_indexes = [33, 133, 160, 144, 158, 153] 
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

#Begins recording
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.75, min_tracking_confidence = 0.75) as face_mesh:
    
    create_file('PupilData.csv')
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
time.sleep(1)
start_time = time.time()
pad_prcnt = 0.20

with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.75, min_tracking_confidence = 0.75) as face_mesh:
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

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            #Ignoring the center because it is unneccessary in this theory
            left_points = [landmarks.landmark[i] for i in LEFT_IRIS[1:]]
            right_points = [landmarks.landmark[i] for i in RIGHT_IRIS[1:]]

            #X and Y values for left eye
            l_xs = [int(p.x * w) for p in left_points]
            l_ys = [int(p.y * h) for p in left_points]

            #X and Y vaues for right eye
            r_xs = [int(p.x * w) for p in right_points]
            r_ys = [int(p.y * h) for p in right_points]
            
            #Finding the extreme points because it is going to be a quadrilateral so better to avoid white part of eyes
            Xmax_L, Xmin_L= max(l_xs), min(l_xs)
            Ymax_L, Ymin_L= max(l_ys), min(l_ys)
            Xmax_R, Xmin_R= max(r_xs), min(r_xs)
            Ymax_R, Ymin_R= max(r_ys), min(r_ys)

            #25% padding calcualted to be added for pupil zone
            l_pad_w = int((Xmax_L - Xmin_L)*pad_prcnt)
            l_pad_h = int((Ymax_L - Ymin_L)*pad_prcnt)
            r_pad_w = int((Xmax_R - Xmin_R)*pad_prcnt)
            r_pad_h = int((Ymax_R - Ymin_R)*pad_prcnt)

            #Determining pupil zone
            Lpupil_zone = flipped_frame[Ymin_L + l_pad_h : Ymax_L - l_pad_h , Xmin_L + l_pad_w : Xmax_L - l_pad_w]
            Rpupil_zone = flipped_frame[Ymin_R + r_pad_h : Ymax_R - r_pad_h , Xmin_R + r_pad_w : Xmax_R - r_pad_w]

            #Concatenation requirements
            if Lpupil_zone.size > 0 and Rpupil_zone.size > 0:

                #Sizing both eyes the same size for concatenation
                L_res = cv2.resize(Lpupil_zone, (20, 20))
                R_res = cv2.resize(Rpupil_zone, (20, 20))

                #Combine the zones
                combined_pupils = np.concatenate((L_res, R_res), axis=1)

                #Determining pupil intensity
                #Seperating pixels based on colours 
                b_chan = combined_pupils[:, :, 0] 
                g_chan = combined_pupils[:, :, 1] 
                r_chan = combined_pupils[:, :, 2] 

                #ITU-R 601 formula used
                intensity_matrix = 0.299 * r_chan + 0.587 * g_chan + 0.114 * b_chan
                avg_intensity = np.mean(intensity_matrix)
                cv2.putText(flipped_frame, str(round(avg_intensity,2)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

            L_center_coord = landmarks.landmark[LEFT_IRIS[0]]
            L_center_point = (int(L_center_coord.x*w), int(L_center_coord.y*h))
            R_center_coord = landmarks.landmark[RIGHT_IRIS[0]]
            R_center_point = (int(R_center_coord.x*w), int(R_center_coord.y*h))
            L_circum_point = (Xmax_L - l_pad_w, Ymax_L - l_pad_h)
            R_circum_point = (Xmax_R - r_pad_w, Ymax_R - r_pad_h)

            L_rad = int(math.dist(L_center_point, L_circum_point))
            R_rad = int(math.dist(R_center_point, R_circum_point))

            # Calculate Pupil Intensity
            if combined_pupils.size > 0:
                #Draw the circle
                cv2.circle(flipped_frame, L_center_point, L_rad, (0,255, 0), 1)
                cv2.circle(flipped_frame, R_center_point, R_rad, (0,255, 0), 1)
                #Draw the boxes
                #cv2.rectangle(flipped_frame,(Xmin_L + l_pad_w, Ymin_L + l_pad_h), (Xmax_L - l_pad_w, Ymax_L - l_pad_h),(0, 0, 255), 1)
                #cv2.rectangle(flipped_frame,(Xmin_R + r_pad_w, Ymin_R + r_pad_h), (Xmax_R - r_pad_w, Ymax_R - r_pad_h),(0, 0, 255), 1)

            time_since_start = time.time() - start_time 

            write_to_file('PupilData.csv', time_since_start, avg_intensity)

        cv2.imshow("Pupil Tracker", flipped_frame)

        if cv2.waitKey(1) & 0xff == ord('x'):
            break
                

