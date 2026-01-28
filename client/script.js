document.addEventListener("DOMContentLoaded", () => {

  console.log("script.js loaded successfully");

  const btn = document.getElementById("predictBtn");

  if (!btn) {
    console.error("Predict button not found");
    return;
  }

  btn.addEventListener("click", async () => {
    console.log("Predict button clicked");

    const data = {
      gender: document.getElementById("gender").value,
      age: document.getElementById("age").value,
      hypertension: document.getElementById("hypertension").value,
      heart_disease: document.getElementById("heart_disease").value,
      ever_married: document.getElementById("ever_married").value,
      work_type: document.getElementById("work_type").value,
      Residence_type: document.getElementById("Residence_type").value,
      avg_glucose_level: document.getElementById("avg_glucose_level").value,
      bmi: document.getElementById("bmi").value,
      smoking_status: document.getElementById("smoking_status").value
    };

    console.log("Sending data:", data);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      const result = await res.json();
      console.log("Response:", result);

      if (result.error) {
        document.getElementById("result").innerHTML =
          "Error: " + result.error;
        return;
      }

      document.getElementById("result").innerHTML = `
        Risk Level: ${result.risk_level}<br>
        Stroke Probability: ${(result.stroke_probability * 100).toFixed(2)}%
      `;
    } catch (err) {
      console.error(err);
      document.getElementById("result").innerHTML =
        "Server not reachable";
    }
  });
});
