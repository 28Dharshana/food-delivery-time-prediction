from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np

# Load the LSTM model
model = load_model('food_delivery_model.keras')   # <-- changed .h5 to .keras

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # simple input form

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    rating = float(request.form['rating'])
    distance = int(request.form['distance'])

    features = np.array([[age, rating, distance]])
    features = features.reshape((features.shape[0], features.shape[1], 1))   # reshape for LSTM!

    prediction = model.predict(features)
    predicted_time = prediction[0][0]

    return render_template('index.html', prediction_text=f'Predicted Delivery Time: {predicted_time:.2f} minutes')

if __name__ == "__main__":
    app.run(debug=True)
