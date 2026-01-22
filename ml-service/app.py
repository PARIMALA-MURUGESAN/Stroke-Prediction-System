from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load artifacts
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
FEATURES = joblib.load("features.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {k.lower(): v for k, v in request.json.items()}
        print("Received:", data)

        required = [
            "gender", "age", "hypertension", "heart_disease",
            "ever_married", "work_type", "residence_type",
            "avg_glucose_level", "bmi", "smoking_status"
        ]

        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Create DataFrame with correct feature names
        input_df = pd.DataFrame([{
            "gender": data["gender"],
            "age": data["age"],
            "hypertension": data["hypertension"],
            "heart_disease": data["heart_disease"],
            "ever_married": data["ever_married"],
            "work_type": data["work_type"],
            "Residence_type": data["residence_type"],
            "avg_glucose_level": data["avg_glucose_level"],
            "bmi": data["bmi"],
            "smoking_status": data["smoking_status"]
        }])

        # Reorder columns
        input_df = input_df[FEATURES]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict probability
        prob = model.predict_proba(input_scaled)[0][1]

        risk = "High" if prob >= 0.20 else "Low"

        return jsonify({
            "risk_level": risk,
            "stroke_probability": round(float(prob), 4)
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
