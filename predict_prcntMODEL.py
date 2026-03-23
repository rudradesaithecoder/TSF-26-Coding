import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# 1. Load and Feature Engineering
df = pd.read_csv('PatientsFile2.csv')
df['Blink_Ratio'] = df['Blink Rate (bpm)'] / df['Blink Duration (ms)']
df['Blink_Product'] = df['Blink Rate (bpm)'] * df['Blink Duration (ms)']

X = df[['Blink Rate (bpm)', 'Blink Duration (ms)', 'Blink_Ratio', 'Blink_Product']]
y = df['Label']

# 2. The "Three-Way" Split
# First, split into Training (80%) and Test (20%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# Second, split Training into actual Train (90% of 80%) and Validation (10% of 80%)
# The Validation set is the 'exam' the AI uses to grade itself via AUC
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.10, stratify=y_temp, random_state=42)

# 3. Scaling (Crucial: Scale all three sets)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Pro XGBoost Model with AUC
model = xgb.XGBClassifier(
    n_estimators=1000,           # More trees, but early stopping will cut it short
    max_depth=6,                # Big picture thinking
    learning_rate=0.1,         # Slow and steady learning
    gamma=1,                    # Only split if it's a good move
    subsample=0.8,              # Randomly use 80% of rows to prevent memorization
    colsample_bytree=0.8,       # Randomly use 80% of features      
    eval_metric='auc',          # THE CHANGE: Grading based on AUC
    early_stopping_rounds=25,   # Stop if AUC doesn't improve for 15 trees 
    random_state=42
)

# 5. Fitting with Early Stopping
model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=True                # Watch the AUC climb in the console
)

# 6. Evaluation
y_probs = model.predict_proba(X_test_scaled)[:, 1]
custom_threshold = 0.42
custom_predictions = (y_probs >= custom_threshold).astype(int)

# Get the importance by 'gain' specifically
importance_gain = model.get_booster().get_score(importance_type='gain')
print("Feature Importance by GAIN:", importance_gain)

print("\n--- Final Results ---")
print(f"Total Model AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
print(f"Classification Report (Threshold {custom_threshold}):")
print(classification_report(y_test, custom_predictions, target_names=['Healthy', 'PD Risk']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, custom_predictions))