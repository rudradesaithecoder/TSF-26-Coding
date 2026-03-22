'''import pandas as pd
import numpy as np
import os
import csv

if os.path.exists('PatientsFile.csv'):
    os.remove('PatientsFile.csv')

hc_rate_mean = 16.90 # Based on Guoy
hc_rate_std = 17.06 # Based on Vasudevan
hc_dur_mean = 55.05 #Based on Vasudevan
hc_dur_std = 8.10 #Based on Vasudevam


corr = -0.21
pd_rate_mean = 10.8 #Based on Gouy
pd_rate_std = 8.92 #based on Vasudevan
pd_dur_mean = 63.25 #Based on Vasudevan
pd_dur_std = 15.74 #based on Vasudevan

number = int(input("Enter the number of datasets you want for each: "))

#Generating random numbers based on paper's 

#Healthy Control numbers 
hc_rate = np.random.normal(hc_rate_mean, hc_rate_std, number)
hc_dur = np.random.normal(hc_dur_mean, hc_dur_std, number)

#Clipping Healthy Control numbers
hc_rate = np.clip(hc_rate, 5.0, None)
hc_dur = np.clip(hc_dur, 40.0, None)

hc_label = [0]*number

#Creating the PD Rate and Clipping
pd_rate = np.random.normal(pd_rate_mean, pd_rate_std, number)

#Creating the PD Duration using the correlation
pd_dur = pd_dur_mean + (corr*(pd_rate - pd_rate_mean)) + np.random.normal(0,10,number)

#Clipping PD numbers
pd_rate = np.clip(pd_rate, 2.0, None) #Nishikawa extremely low clipping
pd_dur = np.clip(pd_dur, 30.0, None)

#PD_Labels
pd_label = [1]*number

#Combining all the patients
blink_rates = np.concatenate([hc_rate,pd_rate])
durations = np.concatenate([hc_dur,pd_dur])
labels = np.concatenate([hc_label,pd_label])



data_file = {
    'Blink Rate (bpm)':  blink_rates,
    'Blink Duration (ms)': durations,
    'Label': labels
}


df = pd.DataFrame(data_file)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('PatientsFile.csv', index=False)
'''



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
print("Your new training file is: master_training_purified.csv")'''