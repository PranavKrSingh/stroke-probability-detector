from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

# Flask app, explicitly pointing to "template" folder
app = Flask(__name__, template_folder='template')

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to model and scaler
scaler_path = os.path.join(BASE_DIR, 'Models', 'scaler.pkl')
model_path = os.path.join(BASE_DIR, 'Models', 'dt.sav')

# Load scaler and model once at startup
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

dt = joblib.load(model_path)


@app.route("/")
def index():
    return render_template("home.html")


@app.route('/about')
def about():
    return render_template('about.html')


@app.route("/result", methods=['POST', 'GET'])
def result():
    try:
        # Get and validate gender
        gender_str = request.form['gender'].lower()
        if gender_str not in ['male', 'female']:
            raise ValueError("Invalid gender value")

        gender = 1 if gender_str == 'male' else 0

        # Collect inputs
        age = int(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        work_type = int(request.form['work_type'])
        Residence_type = int(request.form['Residence_type'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])
        smoking_status = int(request.form['smoking_status'])

        # Create input array
        x = np.array([gender, age, hypertension, heart_disease, ever_married,
                      work_type, Residence_type, avg_glucose_level, bmi,
                      smoking_status]).reshape(1, -1)

        # Scale inputs
        x = scaler.transform(x)

        # Predict
        Y_pred = dt.predict(x)

        # Render result page
        if Y_pred == 0:
            return render_template('nostroke.html')
        else:
            return render_template('stroke.html')

    except ValueError as e:
        return render_template('error.html', message=str(e))
    except Exception as e:
        return render_template('error.html', message=f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's port or default 5000
    app.run(host="0.0.0.0", port=port, debug=True)
