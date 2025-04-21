from flask import Flask, request, render_template
from keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model('food_delivery_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    rating = float(request.form['rating'])
    distance = float(request.form['distance'])
    
    input_data = np.array([[age, rating, distance]])
    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))  # Reshape for LSTM
    prediction = model.predict(input_data)
    
    output = prediction[0][0]
    return render_template('index.html', prediction_text=f'Estimated Delivery Time: {output:.2f} minutes')

if __name__ == "__main__":
    app.run(debug=True)

