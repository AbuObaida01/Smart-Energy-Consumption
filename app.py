# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model
model_path = 'linear_regression_energy.pkl'
with open(model_path, 'rb') as file:
    model = joblib.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get form values
        features = [float(x) for x in request.form.values()]
        final_features = np.array([features])   # 2D array
        prediction = model.predict(final_features)

        output = round(prediction[0], 3)  # regression output
        return render_template('index.html',
                               prediction_text=f'Predicted Global Active Power: {output} kW')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)