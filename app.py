from flask import Flask, request, jsonify
import pickle
import pandas as pd


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

# Load the pre-trained pipeline
with open('spaceship_titanic.pkl', 'rb') as f:
    final_pipeline_lgb = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)



@app.route("/")
def index():
    return '''
    
    POST /predict works
    curl --location 'https://spaceship-prediction-953e7e237ee4.herokuapp.com/predict' \
--header 'Content-Type: application/json' \
--data '{
    "features": [
        ["0029_01", "Europa", true, "B/2/P", "55 Cancri e", 21.0, false, 0.0, 0.0, 0.0, 0.0, 0.0, "Aldah Ainserfle"],
        ["0029_01", "Europa", true, "B/2/P", "55 Cancri e", 21.0, false, 0.0, 0.0, 0.0, 0.0, 0.0, "Aldah Ainserfle"]
    ],
    "threshold": 0.96
}
'''

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.json
    matrix = data['features']
    threshold = data['threshold']
    if not isinstance(matrix[0], list):
        matrix = [matrix]

    X_test = pd.DataFrame(matrix, columns=column_names).reset_index(drop=True)

    y_proba_new = final_pipeline_lgb.predict_proba(X_test)[:, 1]
    y_pred_new = (y_proba_new >= threshold).astype(int)

    response = {
        'probability': y_proba_new.tolist(),
        # Converts array to list for JSON serialization
        'prediction': y_pred_new.tolist()
        # Converts array to list for JSON serialization
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)


