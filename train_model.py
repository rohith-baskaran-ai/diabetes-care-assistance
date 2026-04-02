import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import joblib 
import os 

#Load Data 
print("Loading Dataset...")

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

df = pd.read_csv(url, names=columns)
print(f"Dataset shape: {df.shape}" )
print(f"Diabetic: {df.Outcome.sum()} | Non-diabetic: {(df['Outcome']==0).sum()}")

#Handle Zeros
#In the dataset 0 means missing for these columns

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col].replace(0, df[col].median())
    print(f"Replaced 0s in {col} with median: {df[col].median():.1f}")


#Feature + Target
X = df.drop('Outcome', axis=1)
Y = df['Outcome']

feature_names = X.columns.tolist()
print(f"\nFeatures: {feature_names}")

#Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

#Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

#Train
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, Y_train)
print("Training complete!")

#Evaluate
Y_pred = model.predict(X_test_scaled)
acc = accuracy_score(Y_test, Y_pred)

print(f"\nAccuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred,
      target_names=['Non-diabetic', 'Diabetic']))


# Feature importance
print("\nFeature Importance:")
importances = model.feature_importances_
for feat, imp in sorted(zip(feature_names, importances),
                         key=lambda x: x[1], reverse=True):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<30} {bar} {imp:.4f}")

# ─── SAVE ───────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(model,  'model/diabetes_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

# Save feature names for UI
import json
with open('model/feature_names.json', 'w') as f:
    json.dump(feature_names, f)

print("\nModel saved to model/diabetes_model.pkl")
print("Scaler saved to model/scaler.pkl")
print("Feature names saved to model/feature_names.json")