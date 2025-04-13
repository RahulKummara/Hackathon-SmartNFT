from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model_data = joblib.load("models/nft_pricing_model.pkl")
model = model_data['Model']
scaler = model_data['Scaler']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get inputs from the form (all 10 features)
    volume_usd = float(request.form['volume_usd'])
    sales = int(request.form['sales'])
    floor_price_usd = float(request.form['floor_price_usd'])
    owners = int(request.form['owners'])
    selected_category = request.form['category']

    # Map the category to one-hot encoding
    category_encoded = [0] * 6
    categories = ["Image", "Photography", "Music", "3D", "Street Art", "PFP"]
    category_index = categories.index(selected_category)
    category_encoded[category_index] = 1
    
    # Combine the numeric inputs with the one-hot encoded category
    input_features = [
        volume_usd,
        sales,
        floor_price_usd,
        owners,
        *category_encoded
    ]
    
    # Ensure all 42 features are included (this is just an example)
    # Example: Adding dummy features to match the expected 42 features
    # Modify this based on your actual model feature structure
    full_input = input_features + [0] * (42 - len(input_features))  # Add zeros to match 42 features
    
    # Scale and predict
    scaled_input = scaler.transform([full_input])  # Scaling the input data
    prediction = model.predict(scaled_input)[0]

    # Display the prediction result
    return render_template('index.html', prediction=round(prediction, 2),
                           volume_usd=volume_usd, sales=sales,
                           floor_price_usd=floor_price_usd, owners=owners,
                           selected_category=selected_category)

if __name__ == "__main__":
    app.run(debug=True)
