import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
import os

#Bottom Pause Diagram (Sent to Jai)
'''df = pd.read_csv('voluntary_clinical_training_humanoid.csv')
sns.boxplot(x='Label', y='Pause Time', data=df, palette='Set2')
plt.title("Eyelid 'Bottom Pause' Duration")
plt.xticks([0, 1], ['Healthy', 'Parkinson\'s'])
plt.ylabel("Time at Maximum Closure (ms)")
plt.show()'''


# Splining Data
'''raw_times = np.array([0, 0.033, 0.066, 0.099, 0.132, 0.165]) # 30 FPS timestamps
raw_ear = np.array([0.51, 0.32, 0.08, 0.27, 0.47, 0.5])    # EAR values
cs = CubicSpline(raw_times, raw_ear)
fine_times = np.linspace(0, 0.165, 100) # 1000 Hz equivalent

plt.plot(raw_times, raw_ear, 'ro', label='Webcam (30 FPS)')
plt.plot(fine_times, cs(fine_times), 'b-', label='Spline (1000 Hz)')
plt.title("Virtual High-Speed Reconstruction")
plt.ylabel("Eye Aspect Ratio (EAR)")
plt.legend()
plt.show()'''

# Amplitude and Closing Velocity
'''df = pd.read_csv('clinical_training_noisy_V4.csv')
#df = pd.read_csv('voluntary_clinical_training.csv')
plt.scatter(df[df['Label']==0]['Amplitude'], df[df['Label']==0]['Closing Vel'], alpha=0.4, label='Healthy')
plt.scatter(df[df['Label']==1]['Amplitude'], df[df['Label']==1]['Closing Vel'], alpha=0.4, label='PD')
plt.title("The Kinematic Main Sequence of Spontaneous Blinking")
plt.xlabel("Blink Amplitude (EAR)")
plt.ylabel("Closing Velocity (EAR/ms)")
plt.legend()
plt.show()'''



'''# Load the data your script just created
df = pd.read_csv('clinical_training_noisy_V4.csv')

# 1. Distribution of Closing Velocities (How 'snappy' were the blinks?)
plt.figure(figsize=(8, 4))
sns.histplot(df['Closing Vel'], kde=True, color='blue')
plt.title("Closing Velocity Training Data")
plt.savefig('my_velocity_dist.png')'''


#Data Integrity Pie Chart
'''if os.path.exists('report.csv'):
    df = pd.read_csv('report.csv')
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
    (df['Dur Symmetry'] > 0.2) & (df['Dur Symmetry'] < 5.0)
]
    

    # Calculating the data removed
    final_rows = len(df_results)
    prcnt_removed = round(((initial_rows-final_rows+1)/initial_rows)*100, 2)
    print(f"Percentage Data Removed: {prcnt_removed}%")

labels = ['Valid Blinks', 'Hardware Artifacts/Noise']
sizes = [final_rows+2, initial_rows - final_rows-2]
colors = ['#66b3ff','#ff9999']
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=(0.1, 0))
plt.title("Data Integrity & Artifact Rejection")
plt.show()'''



'''import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score, 
                             accuracy_score, roc_curve, auc, confusion_matrix)

# 1. Load the Clinical Training Data
df = pd.read_csv('clinical_training_noisy_V4.csv')

features = ['Baseline', 'Amplitude', 'Closing Vel', 'Opening Vel', 
            'Closing Dur', 'Opening Dur', 'Pause Time', 
            'Vel Symmetry', 'Dur Symmetry']

X = df[features]
y = df['Label']

# 2. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Clinical XGBoost Model
counter = y.value_counts()
ratio = counter[0] / counter[1]

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=ratio, 
    eval_metric='auc',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# 4. Predictions for Metrics
y_probs = model.predict_proba(X_test_scaled)[:, 1]
prediction = model.predict(X_test_scaled)

# --- GRAPHING SECTION START ---

# A. ROC CURVE (AUC)
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Model AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Clinical Model: ROC Curve Analysis')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig('clinical_roc_curve.png', dpi=300)
plt.show()

# B. CONFUSION MATRIX (Accuracy Visualization)
cm = confusion_matrix(y_test, prediction)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'PD'], yticklabels=['Healthy', 'PD'])
plt.title(f'Confusion Matrix (Acc: {accuracy_score(y_test, prediction)*100:.2f}%)')
plt.ylabel('Actual Diagnosis')
plt.xlabel('AI Predicted Diagnosis')
plt.savefig('clinical_confusion_matrix.png', dpi=300)
plt.show()

# --- FIX: Re-assign feature names to the model ---
model.get_booster().feature_names = features

# Now plot normally
plt.figure(figsize=(10, 6))
xgb.plot_importance(model, importance_type='gain', max_num_features=10, 
                    title='Biometric Importance (Gini Gain)', xlabel='F-Score', grid=False)
plt.tight_layout()
plt.show()

# --- GRAPHING SECTION END ---

print("Clinical Model Trained Successfully.")
print(f"Test AUC: {roc_auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, prediction))'''