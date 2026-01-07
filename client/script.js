document.addEventListener("DOMContentLoaded", () => {
  const btn = document.getElementById("predictBtn");

  btn.addEventListener("click", async () => {
    const data = {
      gender: Number(document.getElementById("gender").value),
      age: Number(document.getElementById("age").value),
      hypertension: Number(document.getElementById("hypertension").value),
      heart_disease: Number(document.getElementById("heart_disease").value),
      ever_married: Number(document.getElementById("ever_married").value),
      work_type: Number(document.getElementById("work_type").value),
      residence_type: Number(document.getElementById("residence").value),
      avg_glucose_level: Number(document.getElementById("glucose").value),
      bmi: Number(document.getElementById("bmi").value),
      smoking_status: Number(document.getElementById("smoking").value)
    };

    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    const result = await res.json();

    document.getElementById("result").innerHTML = `
      <b>Risk Level:</b> ${result.risk_level}<br>
      <b>Stroke Probability:</b> ${(result.stroke_probability * 100).toFixed(2)}%
    `;
  });
});
