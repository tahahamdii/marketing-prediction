from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load saved encoders
encoders = {
    "Offer Accepted": joblib.load("./models/Offer_Accepted_encoder.joblib"),
    "Reward": joblib.load("./models/Reward_encoder.joblib"),
    "Income Level": joblib.load("./models/Income Level_encoder.joblib"),
    "Overdraft Protection": joblib.load("./models/Overdraft Protection_encoder.joblib"),
    "Credit Rating": joblib.load("./models/Credit Rating_encoder.joblib"),
    "Own Your Home": joblib.load("./models/Own Your Home_encoder.joblib"),
}

# Load saved models
cluster_model = joblib.load("./models/kmeans_model.joblib")
scaler = joblib.load("./models/scaler_average_balance.joblib")
regression_model = joblib.load("./models/xgboost_model_optimized.joblib")

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
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])

    # Prepare cluster features (do not include the 'Cluster_Label' column here yet)
    cluster_features = input_df.copy()

    # Predict the cluster label
    input_df["Cluster_Label"] = cluster_model.predict(cluster_features)

    # Prepare features for regression (drop 'Cluster_Label' for the regression model)
    features_for_regression = input_df.drop(columns=["Cluster_Label"])

    # Make prediction using regression model
    scaled_prediction = regression_model.predict(features_for_regression)

    # Unscale the prediction (reverse the scaling)
    unscaled_prediction = scaler.inverse_transform(np.array(scaled_prediction).reshape(-1, 1)).flatten()[0]

    # Return the result as JSON
    return jsonify({"predicted_average_balance": unscaled_prediction})

if __name__ == '__main__':
    app.run(debug=True)
