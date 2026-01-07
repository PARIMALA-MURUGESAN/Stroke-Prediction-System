import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("stroke_data.csv")
if "id" in df.columns:
    df.drop("id", axis=1)
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
# Convert categorical text to numbers
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

# Drop rows with missing values after mapping
df = df.dropna()

features = [
    'gender', 'age', 'hypertension', 'heart_disease',
    'ever_married', 'work_type', 'Residence',
    'avg_glucose_level', 'bmi', 'smoking_status'
]

X = df[features]
y = df['stroke']

# Balance the data
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Train model
model = GradientBoostingClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "stroke_model.pkl")

print("Model trained correctly with", X.shape[1], "features")
