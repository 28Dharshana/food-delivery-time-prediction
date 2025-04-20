from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    rating = float(request.form['rating'])
    distance = float(request.form['distance'])

    features = np.array([[age, rating, distance]])
    prediction = model.predict(features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f"Predicted Delivery Time: {output} minutes")

if __name__ == "__main__":
    app.run(debug=True)
