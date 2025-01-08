from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load saved encoders
encoders = {
    "Offer Accepted": joblib.load("./content/Offer Accepted_encoder.joblib"),
    "Reward": joblib.load("./content/Reward_encoder.joblib"),
    "Income Level": joblib.load("./content/Income Level_encoder.joblib"),
    "Overdraft Protection": joblib.load("./content/Overdraft Protection_encoder.joblib"),
    "Credit Rating": joblib.load("./content/Credit Rating_encoder.joblib"),
    "Own Your Home": joblib.load("./content/Own Your Home_encoder.joblib"),
}

# Load saved models and scaler
cluster_model = joblib.load("./content/kmeans_model.joblib")
scaler = joblib.load("./content/scaler_average_balance.joblib")
regression_model = joblib.load("./content/xgboost_model_optimized.joblib")

# Function to safely encode categorical features
def safe_encode(encoder, value):
    """
    Safely encode a value using the label encoder.
    If the value is not seen during training, it assigns a default value (e.g., 0).
    """
    if value not in encoder.classes_:
        # Temporarily add the new value for encoding, then remove it
        encoder.classes_ = np.append(encoder.classes_, value)
    return encoder.transform([value])[0]

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input data
    input_data = request.json

    # Validate input data
    required_columns = [
        "Offer Accepted", "Reward", "Income Level", "# Bank Accounts Open",
        "Overdraft Protection", "Credit Rating", "# Credit Cards Held",
        "# Homes Owned", "Household Size", "Own Your Home"
    ]

    for col in required_columns:
        if col not in input_data:
            return jsonify({"error": f"Missing required column: {col}"}), 400

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for col, encoder in encoders.items():
        input_df[col] = input_df[col].apply(lambda x: safe_encode(encoder, x))

    # Assign cluster label using all columns after encoding
    
    prediction_features = input_df.copy()

    # Predict scaled Average Balance
    scaled_prediction = regression_model.predict(prediction_features)

    # Unscale the predicted value
    unscaled_prediction = float(scaler.inverse_transform(np.array(scaled_prediction).reshape(-1, 1)).flatten()[0])

    # Return the result
    return jsonify({"predicted_average_balance": round(unscaled_prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
