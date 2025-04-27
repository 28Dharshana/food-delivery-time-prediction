from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'food_delivery_model.keras')
model = load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        age = float(request.form['age'])
        gender = 1 if request.form['gender'] == 'male' else 0
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smoker'] == 'yes' else 0
        region = request.form['region']
        
        # Encode region
        if region == 'northeast':
            region_code = [1, 0, 0, 0]
        elif region == 'northwest':
            region_code = [0, 1, 0, 0]
        elif region == 'southeast':
            region_code = [0, 0, 1, 0]
        elif region == 'southwest':
            region_code = [0, 0, 0, 1]
        else:
            region_code = [0, 0, 0, 0]  # Default fallback
        
        # Prepare input
        input_data = np.array([[age, gender, bmi, children, smoker] + region_code])
        
        # Predict
        prediction = model.predict(input_data)[0][0]
        
        return render_template('index.html', prediction_text=f'Predicted Delivery Cost: ${prediction:.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error occurred: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
