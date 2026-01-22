import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

df = pd.read_csv("stroke_data.csv")
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)
df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
df["ever_married"] = df["ever_married"].map({"Yes": 1, "No": 0})
df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})

df["smoking_status"] = df["smoking_status"].map({
    "never smoked": 0,
    "formerly smoked": 1,
    "smokes": 2,
    "Unknown": 3
})

df["work_type"] = df["work_type"].map({
    "children": 0,
    "Govt_job": 1,
    "Never_worked": 2,
    "Private": 3,
    "Self-employed": 4
})
df.dropna(inplace=True)
FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status"
]

X = df[FEATURES]
y = df["stroke"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_scaled, y)
print("Coefficients:")
for f, c in zip(FEATURES, model.coef_[0]):
    print(f, round(c, 3))
joblib.dump(model, "stroke_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "features.pkl")

print("✅ Logistic Regression model trained successfully")
print("Number of features:", len(FEATURES))
