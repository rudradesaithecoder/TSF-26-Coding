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
        writer.writerow([TimeStamp, pupil_intensity = 1])

#Timer Starter
start_time = time.time()

#Two Key functions
mp_face_mesh = mp.solutions.face_mesh

#Initialising Key Values
right_indexes = [362, 263, 385, 380, 387, 373]  
left_indexes = [33, 133, 160, 144, 158, 153] 
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

folder_path = 'your_images_folder'
create_file('PupilData.csv')

start_time = time.time()
pad_prcnt = 0.20

with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.75, min_tracking_confidence = 0.75) as face_mesh:
    
    for filename in os.listdir(folder_path):
        if not filename.endswith(('.jpg', '.png')): 
            continue
    
        frame = cv2.imread(os.path.join(folder_path, filename))
        if frame is None:
            continue
            
            flipped_frame = cv2.flip(frame, 1)
            h, w, _ = flipped_frame.shape

            # Convert for MediaPipe
            rgb_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)
            flipped_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)
            flipped_frame.flags.writeable = True
            flipped_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            avg_intensity = -1

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
                        
                    cv2.putText(flipped_frame, str(round(np.mean(intensity_matrix),2)), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

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

                write_to_file('PupilData.csv', time_since_start, np.mean(intensity_matrix))

            cv2.imshow("Pupil Tracker", flipped_frame)

cv2.waitKey(0)            
cv2.destroyAllWindows()
