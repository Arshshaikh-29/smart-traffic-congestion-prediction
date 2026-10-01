from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("traffic_model.pkl")

def classify_traffic(speed):
    if speed < 15:
        return "Severe Congestion"
    if speed < 30:
        return "High Congestion"
    if speed < 45:
        return "Moderate Congestion"
    return "Low Congestion"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        features = pd.DataFrame([{
            "hour": int(data["hour"]),
            "day": int(data["day"]),
            "temperature": float(data["temperature"]),
            "rain": float(data["rain"]),
            "vehicles": int(data["vehicles"]),
            "road_capacity": int(data["road_capacity"]),
            "accident": int(data["accident"])
        }])
        if not 0 <= features.at[0, "hour"] <= 23:
            raise ValueError("Hour must be between 0 and 23.")
        if not 0 <= features.at[0, "day"] <= 6:
            raise ValueError("Day must be between 0 and 6.")
        if features.at[0, "vehicles"] < 0 or features.at[0, "road_capacity"] <= 0:
            raise ValueError("Vehicle count must be non-negative and road capacity positive.")
        if features.at[0, "rain"] < 0:
            raise ValueError("Rainfall cannot be negative.")
        distance = float(data["distance"])
        if distance <= 0:
            raise ValueError("Distance must be greater than zero.")

        speed = max(1.0, float(model.predict(features)[0]))
        return jsonify({
            "speed": round(speed, 2),
            "congestion": classify_traffic(speed),
            "travel_time": round(distance / speed * 60, 2)
        })
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(debug=True)
