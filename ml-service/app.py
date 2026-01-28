from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("stroke_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Encoding for model
        gender_map = {"Male": 1, "Female": 0, "Other": 2}
        married_map = {"Yes": 1, "No": 0}
        residence_map = {"Urban": 1, "Rural": 0}
        work_map = {"Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}
        smoking_map = {"formerly smoked": 1, "never smoked": 0, "smokes": 2, "Unknown": 3}

        features = pd.DataFrame([{
            "gender": gender_map[data["gender"]],
            "age": float(data["age"]),
            "hypertension": int(data["hypertension"]),
            "heart_disease": int(data["heart_disease"]),
            "ever_married": married_map[data["ever_married"]],
            "work_type": work_map[data["work_type"]],
            "Residence_type": residence_map[data["Residence_type"]],
            "avg_glucose_level": float(data["avg_glucose_level"]),
            "bmi": float(data["bmi"]),
            "smoking_status": smoking_map[data["smoking_status"]]
        }])

        prob = model.predict_proba(features)[0][1]
        risk = "High" if prob >= 0.10 else "Low"

        return jsonify({
            "risk_level": risk,
            "stroke_probability": round(float(prob), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
