from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# ==============================
# LOAD MODEL + SCALER
# ==============================
model = pickle.load(open("models/flood_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ==============================
# FEATURE ORDER (MUST MATCH TRAINING)
# ==============================
FEATURES = [
    "latitude",
    "longitude",
    "annual_rainfall",
    "temperature",
    "humidity",
    "river_discharge",
    "water_level",
    "elevation",
    "population_density",
    "infrastructure",
    "historical_floods",
    "cloud_visibility",
    "seasonal_rainfall"
]

# ==============================
# HOME PAGE
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# PREDICTION ROUTE
# ==============================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Collect input
        input_data = [float(request.form[f]) for f in FEATURES]

        # Scale input
        final_input = scaler.transform([input_data])

        # Predict
        prediction = model.predict(final_input)[0]

        # ==============================
        # RESULT LOGIC (IMPROVED UI LEVEL)
        # ==============================
        if prediction == 2:
            prediction_text = "🌊 HIGH FLOOD RISK"
            prediction_class = "high"

        elif prediction == 1:
            prediction_text = "⚠️ MEDIUM FLOOD RISK"
            prediction_class = "medium"

        else:
            prediction_text = "✅ LOW / SAFE AREA"
            prediction_class = "low"

        # Send to HTML
        return render_template(
            "result.html",
            prediction_text=prediction_text,
            prediction_class=prediction_class
        )

    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    app.run(debug=True)