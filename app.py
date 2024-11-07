from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

column_names = [
    "PassengerId",
    "HomePlanet",
    "CryoSleep",
    "Cabin",
    "Destination",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "Name",
]

app = Flask(__name__)

try:
    filename = 'spaceship_titanic.pkl'
    with open(filename, 'rb') as f:
        final_pipeline_lgb = joblib.load(f)
except Exception as e:
    app.logger.error("Error loading model: %s", e)
    final_pipeline_lgb = None


@app.route("/")
def index():
    return '''
    <h1>Spaceship Prediction API</h1>
    <p>POST /predict endpoint is available.</p>
    <p>Example usage:</p>
    <pre>
    curl --location 'https://spaceship-prediction-953e7e237ee4.herokuapp.com/predict' \\
    --header 'Content-Type: application/json' \\
    --data '{
        "features": [
            ["0029_01", "Europa", true, "B/2/P", "55 Cancri e", 21.0, false, 0.0, 0.0, 0.0, 0.0, 0.0, "Aldah Ainserfle"],
            ["0029_01", "Europa", true, "B/2/P", "55 Cancri e", 21.0, false, 0.0, 0.0, 0.0, 0.0, 0.0, "Aldah Ainserfle"]
        ],
        "threshold": 0.96
    }'
    </pre>
    '''


@app.route('/predict', methods=['POST'])
def predict():
    # Ensure the model is loaded
    if final_pipeline_lgb is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or 'features' not in data or 'threshold' not in data:
        return jsonify({
                           "error": "Invalid input format. Please provide 'features' and 'threshold'."}), 400

    matrix = data['features']
    threshold = data['threshold']
    if not isinstance(matrix[0], list):
        matrix = [matrix]

    try:
        X_test = pd.DataFrame(matrix, columns=column_names).reset_index(
            drop=True)
    except ValueError as ve:
        return jsonify({"error": f"Feature matrix format error: {ve}"}), 400

    try:
        y_proba_new = final_pipeline_lgb.predict_proba(X_test)[:, 1]
        y_pred_new = (y_proba_new >= threshold).astype(int)
    except Exception as e:
        app.logger.error("Prediction error: %s", e)
        return jsonify({"error": "Prediction error"}), 500

    response = {
        'probability': y_proba_new.tolist(),
        'prediction': y_pred_new.tolist()
    }
    return jsonify(response)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)

# if __name__ == '__main__':
#     app.run(debug=True)
#
