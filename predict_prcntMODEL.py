import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


#Load the Clinical Training Data
'''df = pd.read_csv('voluntary_clinical_training_humanoid.csv')

# These must exactly match your CSV column names
features = ['Baseline', 'Amplitude', 'Closing Vel', 'Opening Vel', 
            'Closing Dur', 'Opening Dur', 'Pause Time', 
            'Vel Symmetry', 'Dur Symmetry']

X = df[features]
y = df['Label']

#Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Clinical XGBoost Model
# We use scale_pos_weight to help the AI "care" more about the PD cases
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
prediction = model.predict(X_test_scaled)'''

'''#Save the Clinical Models
joblib.dump(model, 'vol_clinical_diagnosis_model.pkl')
joblib.dump(scaler, 'vol_clinical_scaler.pkl')'''

'''print("Clinical Model Trained Successfully.")
print(f"Test AUC: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.4f}")

print("Classification Report:")
print(classification_report(y_test, prediction))

print(f'{accuracy_score(y_test, prediction)*100:.2f}%')'''


#Model 2


df = pd.read_csv('ImprovedPatientsFile2.csv')


features = ['Blink Rate (bpm)', 'Blink Duration (ms)']

X = df[features]
y = df['Label']

# 3. Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Blink Rate XGBoost Model
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
prediction = model.predict(X_test_scaled)

'''#Save
joblib.dump(model, 'rate_duration_model.pkl')
joblib.dump(scaler, 'rate_duration_scaler.pkl')'''

#Performance Report
print("Blink Rate/Duration Model Trained Successfully.")
print(f"Test AUC: {roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, prediction))

print(f"Overall Accuracy: {accuracy_score(y_test, prediction)*100:.2f}%")

