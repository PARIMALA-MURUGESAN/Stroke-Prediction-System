from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("stroke_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Normalize input keys
        data = {k.lower(): v for k, v in request.json.items()}
        print(data)

        required = ['gender','age','hypertension','heart_disease','ever_married',
                    'work_type','residence_type','avg_glucose_level','bmi','smoking_status']

        missing = [k for k in required if k not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Create DataFrame to preserve feature names
        features = pd.DataFrame([{
            'gender': data['gender'],
            'age': data['age'],
            'hypertension': data['hypertension'],
            'heart_disease': data['heart_disease'],
            'ever_married': data['ever_married'],
            'work_type': data['work_type'],
            'Residence_type': data['residence_type'],
            'avg_glucose_level': data['avg_glucose_level'],
            'bmi': data['bmi'],
            'smoking_status': data['smoking_status']
        }])

        prob = model.predict_proba(features)[0][1]

        if prob >= 0.10:
            risk = "High"
        else:
            risk = "Low"

        return jsonify({
            "risk_level": risk,
            "stroke_probability": round(float(prob), 4)
        })

    except Exception as e:
        print("MODEL ERROR:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
