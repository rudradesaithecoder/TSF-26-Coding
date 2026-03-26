'''# DELETE THIS ENTIRE BLOCK:
            if len(ear_buffer) >= 3:
                dt = time_buffer[-1] - time_buffer[-2]
                if dt > 0:
                    current_velocity = (ear_buffer[-1] - ear_buffer[-2]) / dt
                    prev_dt = time_buffer[-2] - time_buffer[-3]
                    prev_velocity = (ear_buffer[-2] - ear_buffer[-3]) / prev_dt
                    crnt_acceleration = (current_velocity - prev_velocity) / dt
                else:
                    current_velocity = 0
                    crnt_acceleration = 0'''

import os

if os.path.exists('clinical__training.csv'):
    os.remove('clinical__training.csv')