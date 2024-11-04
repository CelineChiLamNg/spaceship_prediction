from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the pre-trained pipeline
with open('spaceship_titanic.pkl', 'rb') as f:
    pipeline_lgb = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    X = np.array([data['features']])  # Expecting JSON input with "features" key

    # Make predictions
    prediction = pipeline_lgb.predict(X)
    probability = pipeline_lgb.predict_proba(X)[:, 1]

    # Return the results as JSON
    response = {
        'prediction': int(prediction[0]),
        'probability': float(probability[0])
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
