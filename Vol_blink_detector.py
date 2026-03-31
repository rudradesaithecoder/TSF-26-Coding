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
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


#Importing all the models
vol_diag_model = joblib.load('vol_clinical_diagnosis_model.pkl')
vol_diag_scaler = joblib.load('vol_clinical_scaler.pkl')
scaler = joblib.load('scale_model_V9_experimental.pkl')
model = joblib.load('blink_model_V9_experimental.pkl')

'''det_model = joblib.load('rate_duration_model.pkl')
det_scaler = joblib.load('rate_duration_scaler.pkl')
diag_model = joblib.load('clinical_diagnosis_model.pkl')
diag_scaler = joblib.load('clinical_scaler.pkl')'''

'''scaler.feature_names_in_ = None
model.feature_names_in_ = None
det_scaler.feature_names_in_ = None
det_model.feature_names_in_ = None'''

if os.path.exists('report.csv'):
    os.remove('report.csv')


def get_blink_physics_spline(times, values, min_EAR_during_blink, baseline_EAR):
    try:
        times_arr, indices = np.unique(times, return_index=True)
        values_arr = np.array(values)[indices]
        
        if len(times_arr) < 4: return None
        
        spline = CubicSpline(times_arr, values_arr, bc_type='natural')
        res = 1000 
        hi_res_times = np.linspace(times_arr[0], times_arr[-1], num=int((times_arr[-1]-times_arr[0])*res))
        hi_res_values = spline(hi_res_times)
        velocities = spline(hi_res_times, 1)

        peak_idx = np.argmin(hi_res_values)
        peak_ts_precise = hi_res_times[peak_idx]
        
        diagnostic_threshold = min_EAR_during_blink + 0.85 * (baseline_EAR - min_EAR_during_blink)
        below = hi_res_times[hi_res_values <= diagnostic_threshold]
        
        if len(below) < 2: return None # Safety check

        pause_threshold = min_EAR_during_blink + 0.02
        at_bottom = hi_res_times[hi_res_values <= pause_threshold]
        pause_duration = (at_bottom[-1] - at_bottom[0]) if len(at_bottom) > 1 else 0

        start_t = below[0]
        end_t = below[-1]
        
        closing_time = peak_ts_precise - start_t
        opening_time = end_t - peak_ts_precise
        
        c_vel = abs(np.min(velocities))
        o_vel = np.max(velocities)

        return {
            "duration": (end_t - start_t),
            "start_ts_precise": start_t,  
            "end_ts_precise": end_t,      
            "closing_dur": closing_time,
            "opening_dur": opening_time,
            "peak_closing_velocity": c_vel,
            "peak_opening_velocity": o_vel,
            "pause_duration": pause_duration,
            "amplitude": (baseline_EAR - min_EAR_during_blink),
            "velocity_symmetry": o_vel / c_vel if c_vel != 0 else 0,
            "duration_symmetry": opening_time / closing_time if closing_time > 0 else 0,
            "baseline": baseline_EAR
        }
    except Exception as e:
        return None


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
        writer.writerow(["Timestamp", "EAR", "EAR Delta", "Velocity", "Acceleration", "BlinkLabel"])


def write_to_file(filename, TimeStamp, EAR_Val, del_EAR, velocity, acceleration, Blink_Label):
    with open(filename, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([TimeStamp, EAR_Val, del_EAR, velocity, acceleration,  Blink_Label])

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
blinked_frames = 0
blink_count = 0
blink_start = 0
intersaltatory_intervals = []
blink_durations = []
opening_velocities = []
pause_durations = []
blink_amplitudes = []
symmetry_ratios = []
all_min_EARs = []
instantaneous_rates = []
cumulative_rates = []
vel_symmetry = []
dur_symmetry = []
closing_times = []
opening_times = []
closing_velocities = []
right_indexes = [362, 263, 385, 380, 387, 373]  
left_indexes = [33, 133, 160, 144, 158, 153] 
min_EAR_in_blink = 1.0
prev_EAR = 0
prev_velocity = 0
time_of_min_ear = 0
last_blink_start_timestamp = 0
v_target = 0
accel_target = 0
target_ear = 0

#Latest solution idea (Using buffer)

ear_buffer = []
time_buffer = []
buffer_size = 25 # Stores ~0.83 s of data to capture full blink curve
baseline_ear = 0.3
threshold_start = 0.25

#Add new lists to store the spline-calculated physics
spline_peak_velocities = []
#spline_peak_accelerations = []

 
#Begins recording
cap = cv2.VideoCapture(0)
'''cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
'''
with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:
    
    #The Light requirements - to be modified
    light_crit_low = 40
    light_low = 55
    light_high = 160
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
                alert = "Slightly Low Light: Brighten screen or environment"
                color = (0, 255, 255)
                light_eligible = False
            #Green Condition
            elif light_low < eye_brightness < light_high:
                alert = "PERFECT CONDITIONS: Ready to test"
                color = (0, 255, 0)
                light_eligible = True

            #Orange Condition
            elif light_high < eye_brightness <= light_crit_high:
                alert = "Slightly High Light: May cause error in reflex test"
                color = (0, 165, 255)
                light_eligible = False

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
            
        else:
            cv2.putText(flipped_frame, "FACE NOT FOUND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            stable_time = -1

        cv2.imshow("Lighting Optimisation", flipped_frame)
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
last_blink_end = time.time() 
prev_frame_time = time.time()

#Using FaceMesh class
with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:

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

            face_landmark = result.multi_face_landmarks[0]
        
            #Getting the x and y's of the indexes
            R_XYs = [(face_landmark.landmark[r].x, face_landmark.landmark[r].y) for r in right_indexes]
            L_XYs = [(face_landmark.landmark[l].x, face_landmark.landmark[l].y) for l in left_indexes]
            
            #EAR Calculations
            avg_EAR = (EARratio(R_XYs) + EARratio(L_XYs)) / 2

            current_time_wall = time.time() #"Master Clock"
            time_since_start = current_time_wall - start_time

            ear_buffer.append(avg_EAR)
            time_buffer.append(current_time_wall) # Use the raw wall time here
            
            #Ear buffer list new Method
            if len(ear_buffer) > buffer_size:
                ear_buffer.pop(0)
                time_buffer.pop(0)

            #Time difference (dt)
            current_frame_time = time.time()
            
            #Displacement (EAR Delta)
            del_EAR = avg_EAR - prev_EAR
            prev_EAR = avg_EAR # Update for next frame

            #AI Predictor
            if len(ear_buffer) >= 3:
                # Target is the middle frame (index -2) to allow "Look Ahead" (index -1)
                target_ear = ear_buffer[-2]
                prev_ear = ear_buffer[-3]
                next_ear = ear_buffer[-1]
                
                dt_target = time_buffer[-2] - time_buffer[-3]
                dt_next = time_buffer[-1] - time_buffer[-2]
                
                # Velocity/Accel aligned specifically to the target frame
                v_target = (target_ear - prev_ear) / dt_target if dt_target > 0 else 0
                v_next = (next_ear - target_ear) / dt_next if dt_next > 0 else 0
                accel_target = (v_next - v_target) / dt_next if dt_next > 0 else 0
                
                rel_ear = target_ear / baseline_ear if baseline_ear > 0 else 0
                
                clues_v2 = pd.DataFrame([[rel_ear, v_target, accel_target, prev_ear, next_ear]], 
                                        columns=['EAR_Relative', 'Velocity', 'Acceleration', 'EAR_Prev', 'EAR_Next'])

                scld_clues_v2 = scaler.transform(clues_v2)
                prediction = model.predict(scld_clues_v2)[0]
                current_prediction_time = time_buffer[-2]
            else:
                prediction = 0
                current_prediction_time = current_time_wall

            # Update baseline on open eyes
            if prediction == 0 and len(ear_buffer) > 5:
                baseline_ear = np.mean(ear_buffer[-5:])

            # --- THE BLINK TRACKER ---
            if prediction == 1:
                if blinked_frames == 0:
                    blink_count += 1
                    blink_start = current_prediction_time 
                    time_of_min_ear = current_prediction_time
                    actual_blink_start = current_prediction_time - start_time 
                    min_EAR_in_blink = target_ear 
                
                blinked_frames += 1
                if target_ear < min_EAR_in_blink:
                    min_EAR_in_blink = target_ear
                    time_of_min_ear = current_prediction_time
            
            # --- SPLINE PHYSICS ---
            else:
                if blinked_frames > 0:
                    # Your Splicing Logic: Grab the frames that just occurred
                    # We add +4 to the slice to ensure the "Opening" phase is fully in the buffer
                    num_frames = min(blinked_frames + 6, len(ear_buffer)) 
                    blink_times = list(time_buffer)[-num_frames:]
                    blink_EARs = list(ear_buffer)[-num_frames:]

                    physics = get_blink_physics_spline(blink_times, blink_EARs, min_EAR_in_blink, baseline_ear)

                    if physics:
                        gap_duration = physics["start_ts_precise"] - last_blink_end
                        is_first_blink = len(blink_durations) == 0

                        if gap_duration < 0.05 and not is_first_blink:
                            # MERGE LOGIC (For double blinks)
                            #blink_durations[-1] = (physics["end_ts_precise"] - last_blink_start_timestamp)
                            all_min_EARs[-1] = min(all_min_EARs[-1], min_EAR_in_blink)
                            opening_velocities[-1] = max(opening_velocities[-1], physics["peak_opening_velocity"])
                            pause_durations[-1] += physics["pause_duration"]
                            blink_amplitudes[-1] = max(blink_amplitudes[-1], physics["amplitude"])
                            
                            # Use the correct list names and physics keys
                            vel_symmetry[-1] = (vel_symmetry[-1] + physics["velocity_symmetry"]) / 2
                            dur_symmetry[-1] = (dur_symmetry[-1] + physics["duration_symmetry"]) / 2
                            
                            # Update velocities and durations used in averages
                            spline_peak_velocities[-1] = max(spline_peak_velocities[-1], abs(physics["peak_closing_velocity"]))
                            closing_times[-1] += physics["closing_dur"]
                            opening_times[-1] += physics["opening_dur"]
                        else:
                            # NEW BLINK RECORDING
                            #blink_durations.append(physics["duration"])
                            all_min_EARs.append(min_EAR_in_blink)
                            intersaltatory_intervals.append(max(gap_duration, 0))

                            spline_peak_velocities.append(abs(physics["peak_closing_velocity"]))

                            last_blink_start_timestamp = physics["start_ts_precise"]
                            
                            #inst_rate = 60 / gap_duration if gap_duration > 0 else 0
                            #instantaneous_rates.append(inst_rate)

                            closing_velocities.append(physics["peak_closing_velocity"])
                            opening_velocities.append(physics["peak_opening_velocity"])
                            pause_durations.append(physics["pause_duration"])
                            blink_amplitudes.append(physics["amplitude"])
                            vel_symmetry.append(physics["velocity_symmetry"])
                            dur_symmetry.append(physics["duration_symmetry"])
                            closing_times.append(physics["closing_dur"])
                            opening_times.append(physics["opening_dur"])


                            total_mins = (physics["end_ts_precise"] - start_time) / 60.0
                            #cumulative_rates.append(len(blink_durations) / total_mins if total_mins > 0 else 0)

                        last_blink_end = physics["end_ts_precise"]

                    # Reset for next blink
                    min_EAR_in_blink = 1.0
                    blinked_frames = 0
            
            #Fills CSV File
            write_to_file('blinks.csv', 
                          time_since_start, 
                          target_ear,      # Log the EAR the AI actually processed
                          del_EAR, 
                          v_target,        # Aligned velocity
                          accel_target,    # Aligned acceleration
                          prediction)

            #Blink count on screen for debugging
            cv2.putText(video, str(blink_count), (560, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        else:
             cv2.putText(video, "FACE NOT FOUND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)      
        
        #Temporary timer
        if time.time() - start_time > 60:
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

#Total time calculator
total_time = end_time - start_time

#Blink Summary
if blink_count > 0:
    # Ensure all lists match the same length (protects against list mismatch errors)
    min_length = len(pause_durations)

    report_dict = {
        'Blink ID': range(1, min_length + 1),
        'Bottom Pause (ms)': [d*1000 for d in pause_durations[:min_length]],           #NEW
        'Closing Dur (ms)': [d*1000 for d in closing_times[:min_length]],         #new
        'Dur Symmetry': dur_symmetry[:min_length],
        'Opening Dur (ms)': [d*1000 for d in opening_times[:min_length]],         #new
        'Closing Vel (EAR/ms)': spline_peak_velocities[:min_length],
        'Opening Vel (EAR/ms)': opening_velocities[:min_length],                           #NEW
        'Vel Symmetry': vel_symmetry[:min_length],                                #New
        'Amplitude (EAR)' : blink_amplitudes[:min_length],                              #NEW
        #'Time since last (s)': intersaltatory_intervals[:min_length]
        #'Closing Phase (ms)': [d*1000 for d in closing_durations[:min_length]],
        #'Peak Acceleration': spline_peak_accelerations[:min_length],
        #'Min EAR Closure': all_min_EARs[:min_length]
    }

    df_report = pd.DataFrame(report_dict)
    df_report.to_csv('report.csv', index=False)
    



#Still need to add the weightings

# --- MERGED FINAL CLINICAL REPORT ---

# --- FIXED CLINICAL REPORT SECTION ---
if os.path.exists('report.csv'):
    # Data access
    df = pd.read_csv('report.csv')
    initial_rows = len(df)
    
    # Heuristic Data Removal
    df_results = df[                      # Eliminates impossible blinks
    (df['Bottom Pause (ms)'] < 250) &     # Remove blinks that stayed closed > 0.25s
    (df['Duration (ms)'] > 30) &          # Remove "Micro-jitters" < 80ms
    (df['Amplitude (EAR)'] > 0.05) &      # Remove tiny twitches
    (df['Closing Vel (EAR/ms)'] > 1.0) &  # Remove teleporting eyelid errors
    (df['Duration (ms)'] < 250)           # Remove extremely large duration values
    
]
    # Calculating the data removed
    final_rows = len(df_results)
    prcnt_removed = round(((initial_rows-final_rows)/initial_rows)*100, 2)
    print(f"Percentage Data Removed: {prcnt_removed}%")
    
    if not df_results.empty:
        # Calculate means using the EXACT names defined in report_dict
        avg_amp = df_results['Amplitude (EAR)'].mean()
        avg_c_vel = df_results['Closing Vel (EAR/ms)'].mean()
        avg_o_vel = df_results['Opening Vel (EAR/ms)'].mean()
        avg_c_dur = df_results['Closing Dur (ms)'].mean()
        avg_o_dur = df_results['Opening Dur (ms)'].mean()
        avg_pause = df_results['Bottom Pause (ms)'].mean()
        avg_v_sym = df_results['Vel Symmetry'].mean()
        avg_d_sym = df_results['Dur Symmetry'].mean()

        #Micro Model
        avg_features = [baseline_ear, avg_amp, avg_c_vel, avg_o_vel, avg_c_dur, avg_o_dur, avg_pause, avg_v_sym, avg_d_sym]
        
        #Clinical Prediction
        clinical_cols = ['Baseline', 'Amplitude', 'Closing Vel', 'Opening Vel', 'Closing Dur', 'Opening Dur', 'Pause Time', 'Vel Symmetry', 'Dur Symmetry']
        clinical_input = pd.DataFrame([avg_features], columns=clinical_cols)

        scld_clinical = vol_diag_scaler.transform(clinical_input)
        prob_clinical = vol_diag_model.predict_proba(scld_clinical)[0][1]

        #Output
        print(f"Total Blinks: {len(df_results)} | Risk Score: {(prob_clinical)*100:.1f}%")
