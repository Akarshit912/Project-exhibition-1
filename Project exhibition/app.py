from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load pre-trained model (assuming it's saved as 'house_price_model.pkl')
model = pickle.load(open('house_price_model.pkl', 'rb'))

# Maps for encoding categorical values (Location, House Type)
location_map = {'city_center': 1, 'suburbs': 2, 'rural': 3}
house_type_map = {'detached': 1, 'semi-detached': 2, 'apartment': 3}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract input values from the form
    bedrooms = int(data['bedrooms'])
    area = int(data['area'])
    location = data['location']
    house_type = data['house_type']
    bathrooms = int(data['bathrooms'])
    garage = 1 if data['garage'] == 'yes' else 0
    
    # Encode categorical variables
    location_num = location_map.get(location, 0)  # Default to 0 if not found
    house_type_num = house_type_map.get(house_type, 0)  # Default to 0 if not found
    
    # Prepare input features for the model
    input_features = np.array([[bedrooms, area, location_num, house_type_num, bathrooms, garage]])
    
    # Make prediction using the model
    predicted_price = model.predict(input_features)[0]
    
    return jsonify({'price': round(predicted_price, 2)})

if __name__ == '__main__':
    app.run(debug=True)
