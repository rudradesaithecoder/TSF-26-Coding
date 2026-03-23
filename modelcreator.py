import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold #Added features (2 and 3)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

#Load and Prepare
df = pd.read_csv('master_training_SHARPENED.csv')
X = df[['EAR', 'Velocity', 'Acceleration']]
y = df['BlinkLabel']

# Use stratify=y to ensure the test set has the same % of blinks as the training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

#Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Hyperparameter Tuning (The Improvement)
# We test different 'depths' and 'tree counts' to find the best fit
#Dictionary used for Grid method for best parameters for randomtree
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced', 'balanced_subsample'] # Crucial for imbalance
}

# StratifiedKFold ensures every 'fold' has enough blinks to learn from
cv_strategy = StratifiedKFold(n_splits=5)

#Grid search uses different possibilities of sample splits, class weight, esitmators and max depth
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='f1_macro', # We optimize for F1 (balance of precision/recall) rather than just accuracy
    n_jobs=-1 # Uses all your CPU cores for speed
)

print("Starting Grid Search Tuning...")
grid_search.fit(X_train_scaled, y_train)

#Results for best possible option
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

#Print the importance of each factor in AI's decision-making process
importances = best_model.feature_importances_
for feat, importance in zip(['EAR', 'Velocity', 'Acceleration'], importances):
    print(f"Feature '{feat}' Importance: {importance:.4f}")

#Evaluate the model's performance
predictions = best_model.predict(X_test_scaled)
print("\nFinal Performance Report:")
print(classification_report(y_test, predictions, target_names=['Open', 'Blink']))

print(accuracy_score(y_test, predictions)*100)

'''#Save and dump the model
joblib.dump(best_model, 'blink_model_V7_optimized.pkl')
joblib.dump(scaler, 'scale_model_V7_optimized.pkl')
print("\nOptimized Model and Scaler saved!")
'''