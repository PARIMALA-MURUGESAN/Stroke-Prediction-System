import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("stroke_data.csv")

# Drop id column if exists
if "id" in df.columns:
    df = df.drop("id", axis=1)

# Handle missing values
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

# Encode categorical columns
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

df['smoking_status'] = df['smoking_status'].map({
    'never smoked': 0,
    'formerly smoked': 1,
    'smokes': 2,
    'Unknown': 3
})

df['work_type'] = df['work_type'].map({
    'children': 0,
    'Govt_job': 1,
    'Never_worked': 2,
    'Private': 3,
    'Self-employed': 4
})

# Drop rows with missing values
df = df.dropna()

# Features and target
features = [
    'gender', 'age', 'hypertension', 'heart_disease',
    'ever_married', 'work_type', 'Residence_type',
    'avg_glucose_level', 'bmi', 'smoking_status'
]

X = df[features]
y = df['stroke']

# Balance data using SMOTE
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Train Gradient Boosting model
model = GradientBoostingClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "stroke_model.pkl")

print("Model trained successfully with", X.shape[1], "features")
