import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
'''
#Model 1 data file
#Value for mm average
mm_val = 9.0

if os.path.exists('clinical_training.csv'):
    os.remove('clinical_training.csv')

def get_log_params(mean, std):
    variance = std**2
    mu = np.log(mean**2 / np.sqrt(variance + mean**2))
    sigma = np.sqrt(np.log(1 + (variance / mean**2)))
    return mu, sigma

def generate_full_clinical_data(num_people=200, blinks_per_person=20):
    rows = []
    
    # Healthy: Close Vel 189.2, Open Vel 102.6, Amp 7.4mm, Close Dur 69.8ms, Open Dur 190.7ms
    # PD (OFF): Close Vel 91.9, Open Vel 54.3, Amp 3.7mm, Close Dur 69.9ms, Open Dur 194.5ms
    
    for i in range(num_people):
        is_pd = np.random.choice([0, 1])
        subj_baseline = np.random.uniform(0.28, 0.45) # Random eye size
        
        if is_pd == 0:
            # HEALTHY DISTRIBUTIONS
            c_vel_m, c_vel_s = get_log_params(189.2, 40)
            o_vel_m, o_vel_s = get_log_params(102.6, 25)
            amp_m, amp_s = get_log_params(7.4, 1.5)
            c_dur_m, c_dur_s = get_log_params(69.8, 15)
            o_dur_m, o_dur_s = get_log_params(190.7, 40)
            pause_m, pause_s = get_log_params(44.2, 10)
        else:
            # PD DISTRIBUTIONS (Slower, Smaller, Asymmetric)
            c_vel_m, c_vel_s = get_log_params(91.9, 30)
            o_vel_m, o_vel_s = get_log_params(54.3, 15)
            amp_m, amp_s = get_log_params(3.7, 1.2)
            c_dur_m, c_dur_s = get_log_params(69.9, 20)
            o_dur_m, o_dur_s = get_log_params(194.5, 50)
            pause_m, pause_s = get_log_params(135.1, 40)

        for _ in range(blinks_per_person):
            # Generate raw physics (mm and ms)
            raw_c_vel = np.random.lognormal(c_vel_m, c_vel_s)
            raw_o_vel = np.random.lognormal(o_vel_m, o_vel_s)
            raw_amp = np.random.lognormal(amp_m, amp_s)
            c_dur = np.random.lognormal(c_dur_m, c_dur_s)
            o_dur = np.random.lognormal(o_dur_m, o_dur_s)
            pause = np.random.lognormal(pause_m, pause_s)

            # --- SCALE TO EAR ---
            amp_ear = (raw_amp / mm_val) * subj_baseline
            c_vel_ear = (raw_c_vel / mm_val) * subj_baseline
            o_vel_ear = (raw_o_vel / mm_val) * subj_baseline
            
            # --- CALCULATE SYMMETRY RATIOS  ---
            vel_symmetry = o_vel_ear / c_vel_ear if c_vel_ear != 0 else 1
            dur_symmetry = o_dur / c_dur if c_dur != 0 else 1

            rows.append({
                'Baseline': subj_baseline,
                'Amplitude': amp_ear,
                'Closing Vel': c_vel_ear,
                'Opening Vel': o_vel_ear,
                'Closing Dur': c_dur,
                'Opening Dur': o_dur,
                'Pause Time': pause,
                'Vel Symmetry': vel_symmetry,
                'Dur Symmetry': dur_symmetry,
                'Label': is_pd
            })
            
    return pd.DataFrame(rows)

patients = int(input("Enter number of patients to create data: "))
blinks = int(input("Enter desired number of blinks per patient: "))

df = generate_full_clinical_data(patients, blinks)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('clinical_training.csv', index=False)'''


'''
#Model for blinking (spontaneous)
def get_params(mean, std):
    # Calculate sigma for log-space
    sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
    # Calculate mu for log-space
    mu = np.log(mean) - 0.5 * (sigma**2)
    return mu, sigma

if os.path.exists('PatientsFile2.csv'):
    os.remove('PatientsFile2.csv')

hc_rate_mean = 19.30 # Based on Vasudevan
hc_rate_std = 17.06 # Based on Vasudevan
hc_dur_mean = 55.05 #Based on Vasudevan
hc_dur_std = 8.10 #Based on Vasudevan

#Using function logarithmised data values
hc_rate_mu, hc_rate_sigma = get_params(hc_rate_mean, hc_rate_std)
hc_dur_mu, hc_dur_sigma = get_params(hc_dur_mean, hc_dur_std)


corr = -0.233
pd_rate_mean = 15.28 #Based on Vasudevan
pd_rate_std = 8.92 #based on Vasudevan
pd_dur_mean = 63.25 #Based on Vasudevan
pd_dur_std = 15.74 #based on Vasudevan

#Using function logarithmised data values
pd_rate_mu, pd_rate_sigma = get_params(pd_rate_mean, pd_rate_std)
pd_dur_mu, pd_dur_sigma = get_params(pd_dur_mean, pd_dur_std)

number = int(input("Enter the number of datasets you want for each: "))

#Generating random numbers based on paper's 
hc_rates = np.random.lognormal(hc_rate_mu, hc_rate_sigma, number)
hc_durs = np.random.lognormal(hc_dur_mu, hc_dur_sigma, number)
hc_labels = [0]*number

#Clipping Healthy Control numbers
#hc_rate = np.clip(hc_rate, 5.0, None)
#hc_dur = np.clip(hc_dur, 40.0, None)

#Generate the 'randomness' for Rate (as a Z-score)
z_rates = np.random.normal(0, 1, number)
pd_rates = np.exp(pd_rate_mu + pd_rate_sigma * z_rates)

#Generate the 'randomness' for Duration (linked to the Rate's Z-score)
#Use the correlation to pull the Duration Z-score in the opposite direction
z_dur_noise = np.random.normal(0, 1, number)
z_dur = (corr * z_rates) + (np.sqrt(1 - corr**2) * z_dur_noise)
pd_durs = np.exp(pd_dur_mu + pd_dur_sigma * z_dur)

pd_labels = [1]*number

#Combining all the patients
blink_rates = np.concatenate([hc_rates,pd_rates])
durations = np.concatenate([hc_durs,pd_durs])
labels = np.concatenate([hc_labels,pd_labels])


data_file = {
    'Blink Rate (bpm)':  blink_rates,
    'Blink Duration (ms)': durations,
    'Label': labels
}

df = pd.DataFrame(data_file)
print(f"PD Rate Mean: {df[df['Label']==1]['Blink Rate (bpm)'].mean()}")
print(f"PD Corr: {df[df['Label']==1].corr().iloc[0,1]}")


df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('PatientsFile2.csv', index=False)'''


# Model for Voluntary Blinking Data
'''mm_val = 9.0

if os.path.exists('voluntary_clinical_training.csv'):
    os.remove('voluntary_clinical_training.csv')

def get_log_params(mean, std):
    variance = std**2
    mu = np.log(mean**2 / np.sqrt(variance + mean**2))
    sigma = np.sqrt(np.log(1 + (variance / mean**2)))
    return mu, sigma

def generate_voluntary_clinical_data(num_people=200, blinks_per_person=20):
    rows = []
    
    # Kinematic values based on Table 2 for Voluntary Blinking 
    # Healthy: Close Vel 340.1, Open Vel 154.1, Amp 12.4mm, Pause 44.2ms
    # PD (OFF): Close Vel 321.6, Open Vel 150.3, Amp 11.4mm, Pause 135.1ms
    
    for i in range(num_people):
        is_pd = np.random.choice([0, 1])
        subj_baseline = np.random.uniform(0.28, 0.45) 
        
        if is_pd == 0:
            # HEALTHY CONTROLS 
            c_vel_m, c_vel_s = get_log_params(340.1, 20.1)
            o_vel_m, o_vel_s = get_log_params(154.1, 8.3)
            amp_m, amp_s     = get_log_params(12.4, 0.6)
            c_dur_m, c_dur_s = get_log_params(70.8, 3.4)
            o_dur_m, o_dur_s = get_log_params(168.5, 12.6)
            pause_m, pause_s = get_log_params(44.2, 7.6)
        else:
            # PD (OFF THERAPY) 
            # Note: Vel and Amp are similar to controls, but Pause is significantly higher.
            c_vel_m, c_vel_s = get_log_params(321.6, 27.2)
            o_vel_m, o_vel_s = get_log_params(150.3, 15.8)
            amp_m, amp_s     = get_log_params(11.4, 0.8)
            c_dur_m, c_dur_s = get_log_params(75.2, 5.1)
            o_dur_m, o_dur_s = get_log_params(195.3, 16.5)
            pause_m, pause_s = get_log_params(135.1, 27.3) # Significant bradykinesia [cite: 361]

        for _ in range(blinks_per_person):
            raw_c_vel = np.random.lognormal(c_vel_m, c_vel_s)
            raw_o_vel = np.random.lognormal(o_vel_m, o_vel_s)
            raw_amp   = np.random.lognormal(amp_m, amp_s)
            c_dur     = np.random.lognormal(c_dur_m, c_dur_s)
            o_dur     = np.random.lognormal(o_dur_m, o_dur_s)
            pause     = np.random.lognormal(pause_m, pause_s)

            # SCALE TO EAR
            amp_ear = (raw_amp / mm_val) * subj_baseline
            c_vel_ear = (raw_c_vel / mm_val) * subj_baseline
            o_vel_ear = (raw_o_vel / mm_val) * subj_baseline
            
            vel_symmetry = o_vel_ear / c_vel_ear if c_vel_ear != 0 else 1
            dur_symmetry = o_dur / c_dur if c_dur != 0 else 1

            rows.append({
                'Baseline': subj_baseline,
                'Amplitude': amp_ear,
                'Closing Vel': c_vel_ear,
                'Opening Vel': o_vel_ear,
                'Closing Dur': c_dur,
                'Opening Dur': o_dur,
                'Pause Time': pause,
                'Vel Symmetry': vel_symmetry,
                'Dur Symmetry': dur_symmetry,
                'Label': is_pd
            })
            
    return pd.DataFrame(rows)

# Input and Execution
patients = int(input("Enter number of patients to create voluntary data: "))
blinks = int(input("Enter desired number of blinks per patient: "))

df = generate_voluntary_clinical_data(patients, blinks)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('voluntary_clinical_training.csv', index=False)'''



'''import pandas as pd

# 1. Load your master training data
df = pd.read_csv('refined_training_data.csv')

# 2. Define the "Gold Standard" thresholds
V_THRESH = 5.0
A_THRESH = 100.0

# 3. Group contiguous frames into unique 'events'
df['event_id'] = (df['BlinkLabel'] != df['BlinkLabel'].shift()).cumsum()

# 4. Identify the "Peak Strength" of each blink event
blink_groups = df[df['BlinkLabel'] == 1].groupby('event_id')

# Map the peak velocity and acceleration reached during each blink back to the main dataframe
df['peak_vel_event'] = blink_groups['Velocity'].transform(lambda x: x.abs().max())
df['peak_accel_event'] = blink_groups['Acceleration'].transform(lambda x: x.abs().max())

# 5. PURIFY: Re-label weak blinks to 0 (Open)
# A blink is "Weak" if it fails to reach EITHER the Velocity OR Acceleration limit
weak_blink_filter = (df['BlinkLabel'] == 1) & (
    (df['peak_vel_event'] < V_THRESH) | 
    (df['peak_accel_event'] < A_THRESH)
)

frames_changed = len(df[weak_blink_filter])
df.loc[weak_blink_filter, 'BlinkLabel'] = 0

# 6. Save the new high-quality training file
df_purified = df.drop(columns=['event_id', 'peak_vel_event', 'peak_accel_event'])
df_purified.to_csv('master_training_purified.csv', index=False)

print(f"Purification Complete!")
print(f"Total frames re-labeled from 'Blink' to 'Open': {frames_changed}")
print("Your new training file is: master_training_purified.csv")

'''
'''


df = pd.read_csv('PatientFile2.csv')
df_filtered = df[df['Label'] == 1]
plt.scatter(df_filtered['Blink Duration (ms)'], df_filtered['Blink Rate (bpm)'], marker = 'o', linestyle = '-', color = 'blue', alpha=0.6 )
plt.xlabel('Duration')
plt.ylabel('Rate')
plt.title('Correlation Check')
plt.grid(True)
#plt.axhline(y = 4, color = 'black', linestyle = '--', linewidth = 1)
#plt.axhline(y = -4, color = 'black', linestyle = '--', linewidth = 1)
#plt.fill_between(df_filtered['Timestamp'], -4, 4, color = 'green', alpha = 0.3)
plt.savefig('FirstPlotV8.png')
plt.clf()'''