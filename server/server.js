const express = require("express");
const axios = require("axios");
const path = require("path");

const app = express();
app.use(express.json());

// Serve the client folder
app.use(express.static(path.join(__dirname, "..", "client")));

app.post("/predict", async (req, res) => {
  try {
    const response = await axios.post("http://127.0.0.1:5000/predict", req.body);
    res.json(response.data);
  } catch (err) {
    res.status(500).send("Prediction failed");
  }
});

app.listen(3000, () => {
  console.log("Server running at http://localhost:3000");
});
