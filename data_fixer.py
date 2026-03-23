import pandas as pd
import numpy as np
'''
# 1. Load your "purified" data that has the errors
df = pd.read_csv('master_training_purified.csv')

# 2. Define the "Recovery" Thresholds
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
            
            # THE REVERSE HEURISTIC:
            # 1. Is there a significant spike in Velocity or Acceleration?
            high_kinetic = (abs(df.loc[i, 'Velocity']) > v_recovery) or \
                           (abs(df.loc[i, 'Acceleration']) > a_recovery)
            
            # 2. Is the EAR actually lower than the surrounding frames? (The 'V' dip)
            is_a_dip = (df.loc[i, 'EAR'] < df.loc[i-1, 'EAR']) and \
                       (df.loc[i, 'EAR'] < df.loc[i+1, 'EAR'])
            
            # 3. Was there a change in EAR at all?
            has_movement = abs(df.loc[i, 'EAR Delta']) > 0.005
            
            if high_kinetic and is_a_dip and has_movement:
                # RECLAIM IT: Flip the label to 1
                df.loc[i, 'BlinkLabel'] = 1
                # Also flip a small buffer around it so the 'event' is captured
                df.loc[i-1:i+1, 'BlinkLabel'] = 1
                count += 1
                
    return df, count

# Run the fix
df_fixed, recovered_count = reclaim_lost_blinks(df)

print(f"Heuristic Check Complete!")
print(f"Recovered {recovered_count} blink events that were accidentally labeled 0.")

# 3. Save the actually corrected data
df_fixed.to_csv('master_training_FINAL_FIXED.csv', index=False)




import pandas as pd

# Load the data
df = pd.read_csv('master_training_FINAL_FIXED.csv')

# 1. Fill 1-frame gaps (0s surrounded by 1s)
# Fixes 'flickering' labels
def fill_gaps(series):
    arr = series.values.copy()
    for i in range(1, len(arr) - 1):
        if arr[i] == 0 and arr[i-1] == 1 and arr[i+1] == 1:
            arr[i] = 1
    return arr

df['BlinkLabel'] = fill_gaps(df['BlinkLabel'])

# 2. Sync labels with Velocity (Expansion)
# If a frame is 0 but adjacent to a 1 AND velocity is high, it's part of the blink.
v_threshold = df['Velocity'].abs().quantile(0.85)

def sync_with_velocity(df, v_thresh):
    labels = df['BlinkLabel'].values.copy()
    velocities = df['Velocity'].values
    
    # Forward & Backward expansion
    for i in range(1, len(labels)-1):
        if labels[i] == 0:
            # If the next or previous frame is a blink and the current frame 
            # has high kinetic energy, it's actually part of the blink event.
            if (labels[i-1] == 1 or labels[i+1] == 1) and abs(velocities[i]) > v_thresh:
                labels[i] = 1
    return labels

df['BlinkLabel'] = sync_with_velocity(df, v_threshold)

# Save the final, synchronized training set
df.to_csv('master_training_ULTIMATE_FIXED.csv', index=False)'''

import pandas as pd

df = pd.read_csv('master_training_FINAL_FIXED.csv')

# 1. THE SHARPENER: 
# Only keep the '1' label if the EAR is at least 10% below the person's 'Max EAR'
# This removes the '90% open' frames that are confusing the model.
max_ear = df['EAR'].quantile(0.95)
df['BlinkLabel'] = df.apply(lambda x: 1 if (x['BlinkLabel'] == 1 and x['EAR'] < (max_ear * 0.9)) else 0, axis=1)

# 2. FILL GAPS ONLY
# We only fill 1-frame holes to ensure 'Solid Blocks'
for i in range(1, len(df) - 1):
    if df.loc[i-1, 'BlinkLabel'] == 1 and df.loc[i+1, 'BlinkLabel'] == 1:
        df.loc[i, 'BlinkLabel'] = 1

df.to_csv('master_training_SHARPENED.csv', index=False)