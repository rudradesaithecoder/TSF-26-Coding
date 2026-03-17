import pandas as pd

#Advanced version of decision tree
from sklearn.ensemble import RandomForestClassifier

#For splitting training data and testing data 
from sklearn.model_selection import train_test_split

#For Scaling data to match every eye
from sklearn.preprocessing import StandardScaler

#For accuracy score
from sklearn.metrics import accuracy_score

#Model builder
import joblib

#Reads the CSV
df = pd.read_csv('cleaned_eye_data.csv')

#Determines data to be used as clue and data to be predicted
clue = df[['EAR', 'EAR_Change']]
label = df['state']

#Splits the data
clue_tr, clue_tst, label_tr, label_tst = train_test_split(clue, label, test_size=0.2)

#Scales the data
scaler = StandardScaler()
scld_clue_tr = scaler.fit_transform(clue_tr)
scld_clue_tst = scaler.transform(clue_tst)

#AI Module
model = RandomForestClassifier(n_estimators=100)
model.fit(scld_clue_tr, label_tr)

#Accuracy measure
predictions = model.predict(scld_clue_tst)
score = accuracy_score(label_tst, predictions)

#Accuracy measurement
print(f"The score was {score*100:.2f}%")

joblib.dump(model, "blink_model.pkl")
joblib.dump(scaler, "scale_model.pkl")