import pandas as pd
import numpy as np
'''

df = pd.read_csv('master_training_purified.csv')


# We look for the top 10% of movement intensity
v_recovery = df['Velocity'].abs().quantile(0.90)
a_recovery = df['Acceleration'].abs().quantile(0.90)
ear_drop_min = 0.05 # Even a tiny 5% drop counts if it's fast

def reclaim_lost_blinks(df):
    count = 0
    # We look at windows of data to find the 'V' shape
    for i in range(2, len(df) - 2):
        # Only check rows currently labeled as 0 (Open)
        if df.loc[i, 'BlinkLabel'] == 0:
            
            
            high_kinetic = (abs(df.loc[i, 'Velocity']) > v_recovery) or \
                           (abs(df.loc[i, 'Acceleration']) > a_recovery)
            
            
            is_a_dip = (df.loc[i, 'EAR'] < df.loc[i-1, 'EAR']) and \
                       (df.loc[i, 'EAR'] < df.loc[i+1, 'EAR'])
            
           
            has_movement = abs(df.loc[i, 'EAR Delta']) > 0.005
            
            if high_kinetic and is_a_dip and has_movement:
                # RECLAIM IT: Flip the label to 1
                df.loc[i, 'BlinkLabel'] = 1
                # Also flip a small buffer around it so the 'event' is captured
                df.loc[i-1:i+1, 'BlinkLabel'] = 1
                count += 1
                
    return df, count


df_fixed, recovered_count = reclaim_lost_blinks(df)

print(f"Heuristic Check Complete!")
print(f"Recovered {recovered_count} blink events that were accidentally labeled 0.")


df_fixed.to_csv('master_training_FINAL_FIXED.csv', index=False)







df = pd.read_csv('master_training_FINAL_FIXED.csv')


def fill_gaps(series):
    arr = series.values.copy()
    for i in range(1, len(arr) - 1):
        if arr[i] == 0 and arr[i-1] == 1 and arr[i+1] == 1:
            arr[i] = 1
    return arr

df['BlinkLabel'] = fill_gaps(df['BlinkLabel'])


v_threshold = df['Velocity'].abs().quantile(0.85)

def sync_with_velocity(df, v_thresh):
    labels = df['BlinkLabel'].values.copy()
    velocities = df['Velocity'].values
    
    #Forward & Backward expansion
    for i in range(1, len(labels)-1):
        if labels[i] == 0:
            if (labels[i-1] == 1 or labels[i+1] == 1) and abs(velocities[i]) > v_thresh:
                labels[i] = 1
    return labels

df['BlinkLabel'] = sync_with_velocity(df, v_threshold)

# Save the final, synchronized training set
df.to_csv('master_training_ULTIMATE_FIXED.csv', index=False)'''

import pandas as pd

df = pd.read_csv('master_training_FINAL_FIXED.csv')

max_ear = df['EAR'].quantile(0.95)
df['BlinkLabel'] = df.apply(lambda x: 1 if (x['BlinkLabel'] == 1 and x['EAR'] < (max_ear * 0.9)) else 0, axis=1)


for i in range(1, len(df) - 1):
    if df.loc[i-1, 'BlinkLabel'] == 1 and df.loc[i+1, 'BlinkLabel'] == 1:
        df.loc[i, 'BlinkLabel'] = 1

#df.to_csv('master_training_SHARPENED.csv', index=False)




df['Norm_Velocity'] = df['Velocity'].abs() / df['Amplitude']
df['Percent_Closure'] = df['Amplitude'] / df['Baseline_EAR']
df['Pause_Ratio'] = df['Pause_ms'] / df['Total_Duration_ms']


X = df[['Norm_Velocity', 'Percent_Closure', 'Pause_Ratio', 'Acceleration']]