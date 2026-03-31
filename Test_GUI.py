# GUI Imports
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import winsound
from tkinter.scrolledtext import ScrolledText
import time
import csv
import os

#Blink Detector Imports
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
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


root = tk.Tk()
root.geometry("1530x800")
root.title("EyeDetect")
root.resizable(False, False)

# ensure the root grid cell grows so stacked pages fill the window
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

font1 = ("Verdana", 12)
font2 = ("Verdana", 20)
font3 = ("Verdana", 16)
font4 = ("Verdana", 10)
font5 = ("Verdana", 9)

Start_page = tk.Frame(root)
Question_page = tk.Frame(root)
Calibration_page= tk.Frame(root)
QRT_page = tk.Frame(root)
RVB_page = tk.Frame(root)
Results_page = tk.Frame(root)

for f in (Start_page, Question_page, Calibration_page, QRT_page, RVB_page, Results_page):
    f.grid(row=0, column=0, sticky="nsew")

# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def reset_blink_state():
    global ear_buffer, time_buffer, blink_count, blinked_frames
    global blink_durations, opening_velocities, pause_durations
    global blink_amplitudes, symmetry_ratios

    ear_buffer.clear()
    time_buffer.clear()
    blink_count = 0
    blinked_frames = 0

    blink_durations.clear()
    opening_velocities.clear()
    pause_durations.clear()
    blink_amplitudes.clear()
    symmetry_ratios.clear()

det_model = joblib.load('detection_model_V2.pkl')
det_scaler = joblib.load('detection_scaler_V2.pkl')
scaler_v7 = joblib.load('scale_model_V7_optimized.pkl')
model_v7 = joblib.load('blink_model_V7_optimized.pkl')

vol_diag_model = joblib.load('vol_clinical_diagnosis_model.pkl')
vol_diag_scaler = joblib.load('vol_clinical_scaler.pkl')
scaler_v9 = joblib.load('scale_model_V9_experimental.pkl')
model_v9 = joblib.load('blink_model_V9_experimental.pkl')

scaler_v7.feature_names_in_ = None
model_v7.feature_names_in_ = None
det_scaler.feature_names_in_ = None
det_model.feature_names_in_ = None

# -------------------------------
# SCORE STATE + HELPER FUNCTIONS
# -------------------------------

score_state = {
    "psbs_raw": None,
    "psbs_scaled": None,
    "sleep_score": None,
    "time_score": None,
    "age_score": None,

    "sbr_score": None,      # spontaneous blink biomarker score
    "vbr_score": None,      # voluntary blink biomarker score

    "spon_removed": None,   # % removed from spontaneous test
    "vol_removed": None,    # % removed from voluntary test

    "dqcs_spon": None,
    "dqcs_vol": None,
    "dqcs_final": None,

    "pdrisk": None,         # final blink biomarker score
    "final_score": None
}

def calculate_psbs():
    """
    Prodromal Symptom Burden Score from questionnaire inputs.
    Returns raw score, scaled score, and the component sub-scores.
    """
    sleep_map = {
        "7-9 Hours": 1.0,
        "6-7 Hours": 0.7,
        "5-6 Hours": 0.4,
        "Less than 5 Hours": 0.2
    }

    time_map = {
        "Morning (8-12)": 1.0,
        "Afternoon (12-6PM)": 0.8,
        "Night (6PM - 8AM)": 0.6
    }

    age_map = {
        "Younger than 50 Years": 0.0,
        "50-59": 0.5,
        "60-69": 1.0,
        "70-79": 1.5,
        "80+": 2.0
    }

    sleep_score = sleep_map.get(sleep_var.get(), 0.0)
    time_score = time_map.get(tod_var.get(), 0.0)
    age_score = age_map.get(age_group_var.get(), 0.0)

    raw = (
        5.0 * rem_var.get() +
        4.0 * smell_var.get() +
        4.5 * rigidity_var.get() +
        3.0 * balance_var.get() +
        2.2 * constipation_var.get() +
        1.5 * anxiety_var.get() +
        age_score -
        1.0 * essential_tremor_var.get() -
        3.0 * drug_var.get()
    )

    # Simple scaling to 0-100
    # 23 is a practical max reference for this version of the score
    scaled = max(0.0, min(100.0, (raw / 23.0) * 100.0))

    return raw, scaled, sleep_score, time_score, age_score


def calculate_dqcs(sleep_score, time_score, camera_score, percent_removed):
    """
    Data Quality / Confounder Score.
    Higher is better.
    """
    blink_quality = (1.0 - (percent_removed / 100.0)) ** 2
    dqcs = 100.0 * (
        0.25 * sleep_score +
        0.15 * time_score +
        0.25 * camera_score +
        0.35 * blink_quality
    )
    return max(0.0, min(100.0, dqcs))


def update_final_score():
    """
    Combines blink biomarker score, prodromal symptom score, and quality score.
    """
    if score_state["pdrisk"] is None:
        return
    if score_state["psbs_scaled"] is None:
        return
    if score_state["dqcs_final"] is None:
        return

    final_score = (
        0.70 * score_state["pdrisk"] +
        0.30 * score_state["psbs_scaled"]
    ) * (score_state["dqcs_final"] / 100.0)

    score_state["final_score"] = round(final_score, 2)

if os.path.exists('spon_report.csv'):
    os.remove('spon_report.csv')

def get_blink_physics_spline(times, values, min_EAR_during_blink, baseline_EAR):
    """
    Analyzes the blink using Cubic Spline derivatives for high-precision 
    velocity, acceleration, and duration.
    """
    try:
        times_arr, indices = np.unique(times, return_index=True)
        values_arr = np.array(values)[indices]
        
        if len(times_arr) < 4: 
            return None
        
        # Create the spline
        spline = CubicSpline(times_arr, values_arr, bc_type='natural')
        
        # 1ms Resolution interpolation
        res = 1000 
        hi_res_times = np.linspace(times_arr[0], times_arr[-1], num=int((times_arr[-1]-times_arr[0])*res))
        hi_res_values = spline(hi_res_times)
        
        # Calculate Derivatives (Velocity and Acceleration)
        velocities = spline(hi_res_times, 1)
        accelerations = spline(hi_res_times, 2)

        #Calculating the time at peak
        peak_idx = np.argmin(hi_res_values)
        peak_ts_precise = hi_res_times[peak_idx]
        
         # --- NEW DIAGNOSTIC THRESHOLDS ---
        # Capture 85% of the total movement to see the 'slow start' and 'slow finish'
        diagnostic_threshold = min_EAR_during_blink + 0.70 * (baseline_EAR - min_EAR_during_blink)
        below = hi_res_times[hi_res_values <= diagnostic_threshold]
        
        # The 'Pause' threshold: Time spent at the absolute bottom (0.02 range)
        # PD patients 'stick' at the bottom due to switching delays
        pause_threshold = min_EAR_during_blink + 0.02
        at_bottom = hi_res_times[hi_res_values <= pause_threshold]
        pause_duration = (at_bottom[-1] - at_bottom[0]) if len(at_bottom) > 1 else 0

        # --- SYMMETRY LOGIC ---
        start_t = below[0]
        end_t = below[-1]
        
        closing_time = peak_ts_precise - start_t
        opening_time = end_t - peak_ts_precise
        
        # Avoid division by zero; Ratio = Closing / Opening
        # Healthy is usually < 1.0 (Opening takes longer). PD often approaches 1.0 or higher.
        symmetry_ratio = closing_time / opening_time if opening_time > 0 else 1.0
        
        duration = below[-1] - below[0] if len(below) > 1 else (times_arr[-1] - times_arr[0])
        
        # Precise Duration Logic (10% threshold above minimum)
        strict_threshold = min_EAR_during_blink + 0.10 * (baseline_EAR - min_EAR_during_blink)
        below = hi_res_times[hi_res_values <= strict_threshold]
        
        duration = below[-1] - below[0] if len(below) > 1 else (times_arr[-1] - times_arr[0])
        
        return {
            "duration": duration,
            "peak_closing_velocity": np.min(velocities), # Downward stroke
            "peak_opening_velocity": np.max(velocities), # Upward stroke (Crucial for PD)
            "pause_duration": pause_duration,            # The 'Switching' Delay
            "peak_ts_precise": peak_ts_precise,
            "max_acceleration": np.max(np.abs(accelerations)),
            "start_ts_precise": below[0] if len(below) > 0 else times_arr[0],
            "end_ts_precise": below[-1] if len(below) > 0 else times_arr[-1],
            "amplitude": (baseline_EAR - min_EAR_during_blink), # PD has 'low amplitude'
            "symmetry_ratio": symmetry_ratio  # Symmetry ratio of opening and clsoing
        }
    except:
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


def write_to_file(filename, TimeStamp, EAR_Val = 0 , change = 0, velocity = 0, acceleration = 0,  Blink_Label = 0):
    with open(filename, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([TimeStamp, EAR_Val, change,velocity, acceleration,  Blink_Label])

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

# Latest solution idea (Using buffer)
ear_buffer = []
time_buffer = []
buffer_size = 25 # Store ~0.83 s of data to capture full blink curve
baseline_ear = 0.3
threshold_start = 0.25

# Add new lists to store the spline-calculated physics
spline_peak_velocities = []
#spline_peak_accelerations = []

#------------------------------------------------------------------------------------------
                            # Calibration Recording System
#------------------------------------------------------------------------------------------

def start_cali():
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

    cap.release()
    cv2.destroyWindow("Lighting Optimisation")

#------------------------------------------------------------------------------------------
                             # Spontaneous Blink Recording (SBR) System
#------------------------------------------------------------------------------------------

def start_spon_br():

    #Key function
    mp_face_mesh = mp.solutions.face_mesh

    #Importing all the models
    #vol_diag_model = joblib.load('vol_clinical_diagnosis_model.pkl')
    #vol_scaler = joblib.load('vol_clinical_scaler.pkl')
    det_model = joblib.load('rate_duration_model.pkl')
    det_scaler = joblib.load('rate_duration_scaler.pkl')
    scaler = joblib.load('scale_model_V9_experimental.pkl')
    model = joblib.load('blink_model_V9_experimental.pkl')
    diag_model = joblib.load('clinical_diagnosis_model.pkl')
    diag_scaler = joblib.load('clinical_scaler.pkl')



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
            
            diagnostic_threshold = min_EAR_during_blink + 0.70 * (baseline_EAR - min_EAR_during_blink)
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

    #Stopwatch reset
    start_time = time.time()
    last_blink_end = time.time() 
    prev_frame_time = time.time()

    # Set an initial baseline so the first calculation doesn't divide by zero
    #if 'baseline_ear' not in locals() or baseline_ear == 0:
    #    baseline_ear = 0.3

    cap = cv2.VideoCapture(0)
    # Collect raw data
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    if actual_fps <= 0: actual_fps = 30 
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Normalize
    resNorm = min(actual_h / 1080, 1.0)
    fpsNorm = min(actual_fps / 60, 1.0)

    #Calculate Score
    s_cam = (0.5 * resNorm + 0.5 * fpsNorm)
    print(f"Camera Quality Score: {s_cam:.2f}")


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
                    #Target is the middle frame (index -2) to allow "Look Ahead" (index -1)
                    target_ear = ear_buffer[-2]
                    prev_ear = ear_buffer[-3]
                    next_ear = ear_buffer[-1]
                    
                    dt_target = time_buffer[-2] - time_buffer[-3]
                    dt_next = time_buffer[-1] - time_buffer[-2]
                    
                    #Velocity/Accel for target frame
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
                                blink_durations[-1] = (physics["end_ts_precise"] - last_blink_start_timestamp)
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
                                blink_durations.append(physics["duration"])
                                all_min_EARs.append(min_EAR_in_blink)
                                intersaltatory_intervals.append(max(gap_duration, 0))

                                spline_peak_velocities.append(abs(physics["peak_closing_velocity"]))

                                last_blink_start_timestamp = physics["start_ts_precise"]
                                
                                inst_rate = 60 / gap_duration if gap_duration > 0 else 0
                                instantaneous_rates.append(inst_rate)

                                closing_velocities.append(physics["peak_closing_velocity"])
                                opening_velocities.append(physics["peak_opening_velocity"])
                                pause_durations.append(physics["pause_duration"])
                                blink_amplitudes.append(physics["amplitude"])
                                vel_symmetry.append(physics["velocity_symmetry"])
                                dur_symmetry.append(physics["duration_symmetry"])
                                closing_times.append(physics["closing_dur"])
                                opening_times.append(physics["opening_dur"])


                                total_mins = (physics["end_ts_precise"] - start_time) / 60.0
                                cumulative_rates.append(len(blink_durations) / total_mins if total_mins > 0 else 0)

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
                #cv2.putText(video, str(blink_count), (560, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            else:
                cv2.putText(video, "FACE NOT FOUND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)      
            
            #Temporary timer
            if time.time() - start_time > 95:
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
        min_length = len(blink_durations)
        
        blink_rate = blink_count/total_time

        report_dict = {
            'Blink ID': range(1, min_length + 1),
            'Duration (ms)': [d*1000 for d in blink_durations[:min_length]],
            'Bottom Pause (ms)': [d*1000 for d in pause_durations[:min_length]],           
            'Closing Dur (ms)': [d*1000 for d in closing_times[:min_length]],        
            'Dur Symmetry': dur_symmetry[:min_length],
            'Opening Dur (ms)': [d*1000 for d in opening_times[:min_length]],      
            'Closing Vel (EAR/ms)': spline_peak_velocities[:min_length],
            'Opening Vel (EAR/ms)': opening_velocities[:min_length],                      
            'Vel Symmetry': vel_symmetry[:min_length],                               
            'Amplitude (EAR)' : blink_amplitudes[:min_length],                              
        }

        df_report = pd.DataFrame(report_dict)
        df_report.to_csv('spon_report.csv', index=False)
        

        #Still need to add the weightings

        # --- MERGED FINAL CLINICAL REPORT ---

        # --- FIXED CLINICAL REPORT SECTION ---
        if os.path.exists('spon_report.csv'):
            df = pd.read_csv('spon_report.csv')
            initial_rows = len(df)

            df_results = df[
            #Timing Constraints
            (df['Duration (ms)'] > 30) & (df['Duration (ms)'] < 250) & 
            (df['Bottom Pause (ms)'] < 200) &
            
            #Kinematic Sanity (Velocity vs Amplitude)
            # Closing velocity should be significantly higher than 0 for a real blink
            (df['Closing Vel (EAR/ms)'] > 0.002) & 
            (df['Amplitude (EAR)'] > 0.05) &
            
            #Symmetry Sanity (Avoid 'teleporting' eyelid artifacts)
            (df['Vel Symmetry'] > 0.15) & (df['Vel Symmetry'] < 5.0) &
            (df['Dur Symmetry'] > 0.2) & (df['Dur Symmetry'] < 5.0)]
        
            

            # Calculating the data removed
            final_rows = len(df_results)
            prcnt_removed = round(((initial_rows-final_rows)/initial_rows)*100, 2)
            print(f"Percentage Data Removed: {prcnt_removed}%")

        # Calculating the data removed
        final_rows = len(df_results)
        prcnt_removed = round(((initial_rows-final_rows)/initial_rows)*100, 2)
        print(f"Percentage Data Removed: {prcnt_removed}%")

        score_state["spon_removed"] = prcnt_removed
        score_state["dqcs_spon"] = calculate_dqcs(
            score_state["sleep_score"] or 0.0,
            score_state["time_score"] or 0.0,
            s_cam,
            prcnt_removed
        )

        if not df_results.empty:
            # Calculate means using the EXACT names defined in report_dict
            avg_rate = (blink_count / total_time) * 60
            avg_dur = df_results['Duration (ms)'].mean()
            avg_amp = df_results['Amplitude (EAR)'].mean()
            avg_c_vel = df_results['Closing Vel (EAR/ms)'].mean()
            avg_o_vel = df_results['Opening Vel (EAR/ms)'].mean()
            avg_c_dur = df_results['Closing Dur (ms)'].mean()
            avg_o_dur = df_results['Opening Dur (ms)'].mean()
            avg_pause = df_results['Bottom Pause (ms)'].mean()
            avg_v_sym = df_results['Vel Symmetry'].mean()
            avg_d_sym = df_results['Dur Symmetry'].mean()

            #Macro Model
            det_input = pd.DataFrame([[avg_rate, avg_dur]], 
                                    columns=['Blink Rate (bpm)', 'Blink Duration (ms)'])
            scld_det = det_scaler.transform(det_input)
            prob_rate_dur = det_model.predict_proba(scld_det)[0][1]

            #Micro Model
            avg_features = [baseline_ear, avg_amp, avg_c_vel, avg_o_vel, avg_c_dur, avg_o_dur, avg_pause, avg_v_sym, avg_d_sym]
            
            #Clinical Prediction
            clinical_cols = ['Baseline', 'Amplitude', 'Closing Vel', 'Opening Vel', 'Closing Dur', 'Opening Dur', 'Pause Time', 'Vel Symmetry', 'Dur Symmetry']
            clinical_input = pd.DataFrame([avg_features], columns=clinical_cols)

            scld_clinical = diag_scaler.transform(clinical_input)
            prob_clinical = diag_model.predict_proba(scld_clinical)[0][1]

            #Output
            print(f"Total Blinks: {len(df_results)} | Risk Score: {((prob_rate_dur + prob_clinical)/2)*100:.1f}%")

            # Blink biomarker score from spontaneous test
            spon_blink_score = ((prob_rate_dur + prob_clinical) / 2) * 100.0
            score_state["sbr_score"] = round(spon_blink_score, 2)

            update_final_score()

            score_state["pdrisk"] = round(score_state["sbr_score"], 2) if score_state["sbr_score"] is not None else None
            update_final_score()




             

#------------------------------------------------------------------------------------------
                             # Voluntary Blink Recording (VBR) System
#------------------------------------------------------------------------------------------

def start_vol_br():

    global prev_EAR, baseline_ear, blink_count, blinked_frames, min_EAR_in_blink, time_of_min_ear, last_blink_start_timestamp
    global ear_buffer, time_buffer, blink_durations, intersaltatory_intervals, spline_peak_velocities, spline_peak_accelerations, closing_durations, all_min_EARs, instantaneous_rates

    if os.path.exists('vol_report.csv'):
        os.remove('vol_report.csv')


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
            
            diagnostic_threshold = min_EAR_during_blink + 0.90 * (baseline_EAR - min_EAR_during_blink)
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
    

    # Collect raw data
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    if actual_fps <= 0: actual_fps = 30 
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # Normalize
    resNorm = min(actual_h / 1080, 1.0)
    fpsNorm = min(actual_fps / 30, 1.0)

    #Calculate Score
    s_cam = (0.5 * resNorm + 0.5 * fpsNorm)
    print(f"Camera Quality Score: {s_cam:.2f}")

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

        
    # Initialize timers immediately
    start_time = time.time()
    last_blink_end = time.time() 
    prev_frame_time = time.time()

    # Ensure the baseline has a starting point so the first calculation doesn't fail
    if baseline_ear == 0:
        baseline_ear = 0.3

    # Using FaceMesh class to begin recording immediately
    with mp_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as face_mesh:

        create_file('blinks.csv')
        # Main Detection Loop
        while cap.isOpened():

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

                            scld_clues_v2 = scaler_v9.transform(clues_v2)
                            prediction = model_v9.predict(scld_clues_v2)[0]
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
                    else:
                        cv2.putText(video, "FACE NOT FOUND", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)      
                    
                    #Temporary timer
                    if time.time() - start_time > 35:
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
        df_report.to_csv('vol_report.csv', index=False)
        



    #Still need to add the weightings

    # --- MERGED FINAL CLINICAL REPORT ---

    # --- FIXED CLINICAL REPORT SECTION ---
    if os.path.exists('vol_report.csv'):
        df = pd.read_csv('vol_report.csv')
        initial_rows = len(df)

        df_results = df[
            #Timing Constraints
            (df['Bottom Pause (ms)'] < 200) &
            
            #Kinematic Sanity (Velocity vs Amplitude)
            # Closing velocity should be significantly higher than 0 for a real blink
            (df['Closing Vel (EAR/ms)'] > 0.002) & 
            (df['Amplitude (EAR)'] > 0.05) &
            
            #Symmetry Sanity (Avoid 'teleporting' eyelid artifacts)
            (df['Vel Symmetry'] > 0.15) & (df['Vel Symmetry'] < 5.0) &
            (df['Dur Symmetry'] > 0.2) & (df['Dur Symmetry'] < 5.0)]
        
            

            # Calculating the data removed
        final_rows = len(df_results)
        prcnt_removed = round(((initial_rows-final_rows)/initial_rows)*100, 2)
        print(f"Percentage Data Removed: {prcnt_removed}%")

        # Calculating the data removed
        final_rows = len(df_results)
        prcnt_removed = round(((initial_rows-final_rows)/initial_rows)*100, 2)
        print(f"Percentage Data Removed: {prcnt_removed}%")

        score_state["vol_removed"] = prcnt_removed
        score_state["dqcs_vol"] = calculate_dqcs(
            score_state["sleep_score"] or 0.0,
            score_state["time_score"] or 0.0,
            s_cam,
            prcnt_removed
        )

        # Combine both quality scores
        if score_state["dqcs_spon"] is not None and score_state["dqcs_vol"] is not None:
            score_state["dqcs_final"] = round(
                (score_state["dqcs_spon"] + score_state["dqcs_vol"]) / 2.0, 2
            )
        else:
            score_state["dqcs_final"] = score_state["dqcs_vol"]

        update_final_score()

        
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

            # Blink biomarker score from voluntary test
            score_state["vbr_score"] = round(prob_clinical * 100.0, 2)

            if score_state["sbr_score"] is not None and score_state["vbr_score"] is not None:
                score_state["pdrisk"] = round(
                    0.6 * score_state["sbr_score"] + 0.4 * score_state["vbr_score"], 2
                )

    update_final_score()


        
                  
# -------------------------
# Background image (on Start_page)
# -------------------------
# load & resize to match window (800x800)
img = Image.open("bg1.png").resize((1530, 800))
bg = ImageTk.PhotoImage(img)

bg_label = tk.Label(Start_page, image=bg)
bg_label.image = bg
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
bg_label.lower()   # <- send background to back (important)

# -------------------------
# Start page content
# -------------------------
title = tk.Label(Start_page, text="Eye-Detect - Online Parkinson's Disease Screening tool",
                 font=font2, bg="#ffffff")
# place title at top (we keep it visible above background)
title.pack(pady=(20, 10))

# center container for the form (use pack + a fixed width frame)
center_frame = tk.Frame(Start_page, bg="", width=380)
center_frame.pack(pady=10)            # packs centered horizontally by default

# Participant form inside center_frame: use grid (safe — only used inside center_frame)
name_var = tk.StringVar()
sex_var = tk.StringVar()

lbl_fname = tk.Label(center_frame, text="Full Name", anchor="e")
ent_fname = tk.Entry(center_frame, textvariable=name_var, width=30)

lbl_sex = tk.Label(center_frame, text="Sex (M/F)", anchor="e")
ent_sex = tk.Entry(center_frame, textvariable=sex_var, width=30)

lbl_fname.grid(row=0, column=0, padx=8, pady=6, sticky="e")
ent_fname.grid(row=0, column=1, padx=8, pady=6, sticky="w")
lbl_sex.grid(row=1, column=0, padx=8, pady=6, sticky="e")
ent_sex.grid(row=1, column=1, padx=8, pady=6, sticky="w")

def submit():
    name = name_var.get()
    sex = sex_var.get()

    upper_sex = sex.upper()

    if (name != "" ) and (upper_sex == "M" or upper_sex == "F"):
        print(name, upper_sex)
        start_button.config(state="active")
    
submit_btn = tk.Button(center_frame, text="Submit", command=submit)
submit_btn.grid(row=3, column=0, columnspan=2, pady=(8,0))


# Raise start page
Start_page.tkraise()

# -------------------------
# Example for (calibration) — simple label + back button
# -------------------------

progress_value = tk.IntVar()
progress_value.set(0)

for p in ( Question_page, QRT_page, RVB_page, Results_page):
    bar = ttk.Progressbar(p, orient="vertical",length=300, mode="determinate", variable=progress_value, maximum=100)
    bar.pack(side="right", padx=20, pady=20)

style = ttk.Style()
style.configure("TProgressbar", thickness=20)

def go_to_Question():
   Question_page.tkraise()

def start_calibration1():
    progress_value.set(100/4*1)
    threading.Thread(target=start_cali, daemon=True).start()
    Calibration_page.tkraise()

def go_to_QRT():
    progress_value.set(100/4*2)
    QRT_page.tkraise()

def go_to_RVB():
    progress_value.set(100/4*3)
    RVB_page.tkraise()

def go_to_Results():
    progress_value.set(100/4*4)

    if score_state["final_score"] is None:
        results_text.set(
            "Results are not ready yet.\n\n"
            "Complete the questionnaire, Quiet Rest Test, and Voluntary Blink Test first."
        )
        disclaimer_text.set("")
    else:
        results_text.set(
            f"Blink Biomarker Score: {score_state['pdrisk']:.2f}%\n"
            f"Prodromal Symptom Burden Score: {score_state['psbs_scaled']:.2f}%\n"
            f"Data Quality / Confounder Score: {score_state['dqcs_final']:.2f}%\n"
            f"Final Combined Score: {score_state['final_score']:.2f}%"
        )

        if score_state.get("eye_condition_flag", False):
            disclaimer_text.set(
                "Disclaimer: Your results are not conclusive due to your underlying eye-condition(s), "
                "therefore, it is suggested to visit a doctor for a more conclusive diagnosis."
            )
        else:
            disclaimer_text.set("")

    Results_page.tkraise()

# Start test button (visible and below the form)
start_button = tk.Button(Start_page, text="Go to Questionnaire", font=font1, command=go_to_Question, state="disabled")
start_button.pack(pady=(12, 20))



#-------------------------------------------------------------------------------------------------------------------------------------------
                                                         #Questionnaire
#-------------------------------------------------------------------------------------------------------------------------------------------

Question_lbl = tk.Label(Question_page, font=font2, text = "Questionnaire")
Question_lbl.pack()

Instructions = tk.Label(Question_page, font = font4, text='Check of the following sleep factors and symptoms you currently have.')
Instructions.pack(pady=10)

#Symptom and external factor variables

sleep_var = tk.StringVar(value="none")
tod_var = tk.StringVar(value="none")
age_group_var = tk.StringVar(value="none")
eye_condition_var = tk.StringVar(value="None")

rem_var = tk.IntVar(value=0)
smell_var = tk.IntVar(value=0)
rigidity_var = tk.IntVar(value=0)
balance_var = tk.IntVar(value=0)
constipation_var = tk.IntVar(value=0)
anxiety_var = tk.IntVar(value=0)
essential_tremor_var = tk.IntVar(value=0)
tremor_var = tk.IntVar(value=0)
drug_var = tk.IntVar(value=0)
score_state["eye_condition_flag"] = False

# Creating Eye Condition Checkboxes
eye_condition_lbl = tk.Label(Question_page, font=font1, text="Pre-existing Eye Conditions")
eye_condition_lbl.pack(pady=5)

eye_condition_combo = ttk.Combobox(
    Question_page,
    textvariable=eye_condition_var,
    values=[
        "None",
        "Needs Eye-glasses(Far/Short-sightedness)",
        "Eye strain",
        "Facial Tics",
        "Corneal Abrasion",
        "Trichiasis",
        "Blepharitis",
        "Blepharospasm",
        "Dry Eye Syndrome",
        "2 or more eye conditions"
    
    ],
    state="readonly"
)

eye_condition_combo.set("None")
eye_condition_combo.pack(pady=5)

# Creating Sleep Checkboxes
Sleep_Lbl = tk.Label(Question_page, font=font1, text="Hours Sleep Last Night")
Sleep_Lbl.pack(pady=5)

sleep_combo = ttk.Combobox(
    Question_page,
    textvariable=sleep_var,
    values=["7-9 Hours", "6-7 Hours", "5-6 Hours", "Less than 5 Hours"],
    state="readonly"   # prevents typing
)

sleep_combo.set("Select option")  # placeholder text
sleep_combo.pack(pady=5)

# Creating TOD Checkboxes
TOD_Lbl = tk.Label(Question_page, font=font1, text="Time of Day")
TOD_Lbl.pack(pady=5)

tod_combo = ttk.Combobox(
    Question_page,
    textvariable=tod_var,
    values=["Morning (8-12)", "Afternoon (12-6PM)", "Night (6PM - 8AM)"],
    state="readonly"
)

tod_combo.set("Select option")
tod_combo.pack(pady=5)

# Creating Age Checkboxes
Age_Lbl = tk.Label(Question_page, font=font1, text="Your Age") 
Age_Lbl.pack(pady=5)

age_combo = ttk.Combobox(
    Question_page,
    textvariable=age_group_var,
    values=["Younger than 50 Years", "50-59", "60-69", "70-79", "80+"],
    state="readonly"
)

age_combo.set("Select option")
age_combo.pack(pady=5)

# Creating Symptoms Checkboxes
TOD_Lbl = tk.Label(Question_page, font=font1, text="Symptoms (Check all that apply)") 
TOD_Lbl.pack(pady=5)

tk.Checkbutton(Question_page, text="REM Sleep Behavior Disorder", variable=rem_var).pack()
tk.Checkbutton(Question_page, text="Loss of smell", variable=smell_var).pack()
tk.Checkbutton(Question_page, text="Slow movement/rigidity", variable=rigidity_var).pack()
tk.Checkbutton(Question_page, text="Balance issues", variable=balance_var).pack()
tk.Checkbutton(Question_page, text="Constipation", variable=constipation_var).pack()
tk.Checkbutton(Question_page, text="Constant anxiety/depression", variable=anxiety_var).pack()
tk.Checkbutton(Question_page, text="Tremors", variable=tremor_var).pack()
tk.Checkbutton(Question_page, text="Diagnosed Essential Tremors", variable=essential_tremor_var).pack()
tk.Checkbutton(Question_page, text="Dopamine-blocking drugs", variable=drug_var).pack()

if sleep_var == "Select option" or tod_var == "Select option" or age_group_var == "Select option":
    print("Please complete all fields")

#Inputs Questionnaire Info into a CSV File
def submit_questions():
    sleep = sleep_var.get()
    tod = tod_var.get()
    age_group = age_group_var.get()
    eye_condition = eye_condition_var.get()

    # Store eye-condition flag
    score_state["eye_condition_flag"] = (eye_condition != "None")

    # Calculate and store prodromal symptom score
    raw_psbs, scaled_psbs, sleep_score, time_score, age_score = calculate_psbs()
    score_state["psbs_raw"] = round(raw_psbs, 2)
    score_state["psbs_scaled"] = round(scaled_psbs, 2)
    score_state["sleep_score"] = sleep_score
    score_state["time_score"] = time_score
    score_state["age_score"] = age_score

    symptoms = {
        "REM Sleep Behavior Disorder": rem_var.get(),
        "Loss of smell": smell_var.get(),
        "Slow movement/rigidity": rigidity_var.get(),
        "Balance issues": balance_var.get(),
        "Constipation": constipation_var.get(),
        "Constant anxiety/depression": anxiety_var.get(),
        "Tremors": tremor_var.get(),
        "Diagnosed Essential Tremors": essential_tremor_var.get(),
        "Dopamine-blocking drugs": drug_var.get()
    }

    row = {
        "eye_condition": eye_condition,
        "sleep": sleep,
        "time_of_day": tod,
        "age_group": age_group,
        "psbs_raw": score_state["psbs_raw"],
        "psbs_scaled": score_state["psbs_scaled"],
        **symptoms
    }

    file_exists = os.path.isfile("questionnaire.csv")
    with open("questionnaire.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print("Saved:", row)
    submit_questions_btn.config(state="disabled")

# Submit Button
submit_questions_btn = tk.Button(Question_page, text="Submit", font=font1, command=submit_questions)
submit_questions_btn.pack(pady=30)

ST_button = tk.Button(Question_page, text="Start Calibration", command=start_calibration1, font = font3, state="active")
ST_button.pack()

#--------------------------------------------------------------------------------------------------------------------------------------------
                       # Calibration Page
#--------------------------------------------------------------------------------------------------------------------------------------------

lbl2 = tk.Label(Calibration_page, text="Calibration Test", font=font2)
lbl2.pack(pady=20)

instruction1 = "Instructions: Remove your glasses for the remainder of the test. You can put them on in between tests to read instructions. \n To  help calibrate the system, centre your face in the box.\n The system will then tell you to press 'c' to start the blink detection software. \n Afterwards press the 'Start Testing' button to move on to the first test."
cali_instructions = tk.Label(Calibration_page, text=instruction1, font=font1)
cali_instructions.pack()
btn1 = tk.Button(Calibration_page, text=" Start Testing ", font=font1, command=go_to_QRT)
btn1.pack(side="bottom", pady=70)

canvas=tk.Canvas(Calibration_page, width = 650, height = 650)
canvas.pack(pady=20)

#------------------------------------------------------------------------------------------------------------------------------------        
                                                      # Quiet Rest Test
#------------------------------------------------------------------------------------------------------------------------------------
lbl3 = tk.Label(QRT_page, text="Quiet Rest Test", font=font2)
lbl3.pack(pady=20)

instruction2 = "Instructions: Simply relax and look at a distant object/wall/camera straight ahead of you, keeping your face visible to the camera. \n A 95 second timer will start when you press the 'start test' button, increase your volume. \n Once the timer runs out you will hear a bell sound, at that point you can move on to the next test."
read_instructions = tk.Label(QRT_page, text=instruction2, font=font4)
read_instructions.pack()

next_test2 = tk.Label(QRT_page, text = "PRESS 'Next Test' NOW!", font=font2)

#---------- Bell Sound Logic ---------

def play_sound(bellsound):
    winsound.PlaySound('bellsound.wav', winsound.SND_FILENAME) 

def on_btn_click():
    threading.Thread(target=play_sound, args=('bellsound.wav',),daemon=True).start()
def play():
    reset_blink_state()
    start_timer()
    QRT_page.after(100, lambda: threading.Thread(target=start_spon_br, daemon=True).start())
    QRT_page.after(95000, on_btn_click)
    QRT_page.after(95000,  lambda: next_test2.pack(pady=60))


    

# ---------------- TIMER LOGIC ----------------
# Label to display time
timer_label = tk.Label(QRT_page, text="95", font=font2)
timer_label.pack(pady=50)

def start_timer():
    countdown(95)  # start from 90 seconds

def countdown(time_left):
    if time_left >= 0:
        timer_label.config(text=str(time_left))
        QRT_page.after(1000, lambda: countdown(time_left - 1))

start_timer0 = tk.Button(QRT_page, text="Start Test", font=font1, command=play)
start_timer0.pack(pady=25)

btn2 = tk.Button(QRT_page, text="Next Test", command=go_to_RVB, font=font1)
btn2.pack(side="bottom", pady=50)

#---------------------------------------------------------------------------------------------------------------
                                      # Voluntary Blinking Test
#---------------------------------------------------------------------------------------------------------------

lbl6 = tk.Label(RVB_page, text="Voluntary Blinking Test", font=font2)
lbl6.pack(pady=20)
btn5=tk.Button(RVB_page, text="Results", command=go_to_Results, font=font1)
btn5.pack(side="bottom", pady=50)

instruction5 = "Instructions: In this test, you will voluntarily blink for 35 seconds. \n Once ready, press the start button, look torwards your camera, and blink at a moderate pace compared to rapid blinking. \n Avoid half-blinks and complete your blinks consistently. \n Once the 35 seconds are over you will hear a bell sound, then you can move on to your results."
rbt_instructions = tk.Label(RVB_page, text=instruction5, font=font4)
rbt_instructions.pack()

timer_label1 = tk.Label(RVB_page, text="35", font=font2)
timer_label1.pack(pady=50)

results_lbl = tk.Label(RVB_page, text = "PRESS 'Results' NOW!", font=font2)

def start_timer1():
    countdown1(35)  # start from 35 seconds

def countdown1(time_left):
    if time_left >= 0:
        timer_label1.config(text=str(time_left))
        RVB_page.after(1000, lambda: countdown1(time_left - 1))

def play1():
    reset_blink_state()
    start_timer1()
    RVB_page.after(100, lambda: threading.Thread(target=start_vol_br, daemon=True).start())
    root.after(35000, on_btn_click)
    RVB_page.after(35000, on_btn_click)
    RVB_page.after(35000, lambda: results_lbl.pack(pady=60))


start_test1 = tk.Button(RVB_page, text="Start Test", font=font3, command=play1)
start_test1.pack(pady=50)

#------------------------------------------------------------------------------------------------------------------------------
                                                       # Results
#------------------------------------------------------------------------------------------------------------------------------
lbl7 = tk.Label(Results_page, text="Results", font=font2)
lbl7.pack(pady=20)

disclaimer = tk.Label(Results_page, text="Disclaimer: This data has not been medically tested. \n The test does not give a definitive answers or having Parkinson's Disease or not. \n If you have an existing eye-condition, the results will not be conclusive, it is recommended to talk to a doctor.", font=font1).pack(pady=15)

results_text = tk.StringVar()
results_text.set("Run the questionnaire and both blink tests to generate your scores.")

results_display = tk.Label(
    Results_page,
    textvariable=results_text,
    font=font4,
    justify="left",
    wraplength=900
)
results_display.pack(pady=20)

disclaimer_text = tk.StringVar()
disclaimer_label = tk.Label(
    Results_page,
    textvariable=disclaimer_text,
    font=font4,
    fg="red",
    justify="left",
    wraplength=900
)
disclaimer_label.pack(pady=10)

comprehension_lbl = tk.Label()

# show GUI
root.mainloop()