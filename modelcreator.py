import pandas as pd

#Advanced version of decision tree
from sklearn.ensemble import RandomForestClassifier

#For splitting training data and testing data 
from sklearn.model_selection import train_test_split, cross_val_score

#For Scaling data to match every eye
from sklearn.preprocessing import StandardScaler

#For accuracy score
from sklearn.metrics import accuracy_score

#Classification Report
from sklearn.metrics import classification_report

#Model builder
import joblib

#Reads the CSV
df = pd.read_csv('trainingdata3.csv')

#Determines data to be used as clue and data to be predicted
clue = df[['EAR', 'EAR Delta']]
label = df['BlinkLabel']

#Splits the data
clue_tr, clue_tst, label_tr, label_tst = train_test_split(clue, label, test_size=0.2)

#Scales the data
scaler = StandardScaler()
scld_clue_tr = scaler.fit_transform(clue_tr)
scld_clue_tst = scaler.transform(clue_tst)

#AI Module
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(scld_clue_tr, label_tr)

#Cross Validation
cv_scores = cross_val_score(model, scld_clue_tr, label_tr, cv=5)
print(f"Cross-Validation Mean Score: {cv_scores.mean()*100:.2f}%")

#Accuracy measure
predictions = model.predict(scld_clue_tst)
score = accuracy_score(label_tst, predictions)

#Accuracy output
print(f"The score was {score*100:.2f}%")


print("Detailed Performance Report:")

# This compares the 'Real' labels vs the 'Predicted' labels
print(classification_report(label_tst, predictions, target_names=['Open', 'Blink']))

#Save the model and the scaler
joblib.dump(model, 'blink_model_V6.pkl')
joblib.dump(scaler, 'scale_model_V6.pkl')

print("Model and Scaler saved successfully!")

