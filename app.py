from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model and scaler
xgb_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Pregnancy-specific normal ranges
pregnancy_normal_ranges = {
    "Age": (18, 35),
    "Body Temperature(F)": (97, 99),
    "Heart rate(bpm)": (70, 110),
    "Systolic Blood Pressure(mm Hg)": (65, 140),
    "Diastolic Blood Pressure(mm Hg)": (70, 80),
    "BMI(kg/m 2)": (18.5, 24.9),
    "Blood Glucose(HbA1c)": (0, 42),  # Upper limit for HbA1c
    "Blood Glucose(Fasting hour-mg/dl)": (3.3, 5.1),
}

# Map prediction outcome to risk levels
def risk_level(outcome):
    levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    return levels.get(outcome, "Unknown Risk")

# Explain risk factors
def explain_risk_factors(user_input):
    explanations = []
    for feature, value in user_input.items():
        if feature in pregnancy_normal_ranges:
            low, high = pregnancy_normal_ranges[feature]
            if value < low:
                explanations.append(f"Low {feature} ({value})")
            elif value > high:
                explanations.append(f"High {feature} ({value})")
    return explanations

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/price')
def price():
    return render_template('price.html')

@app.route('/testimonial')
def testimonial():
    return render_template('testimonial.html')

@app.route('/check', methods=['GET'])
def check():
    return render_template('check.html', result=None, explanations=None)

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Get form inputs
        age = float(request.form['Age'])
        height = float(request.form['height'])  
        weight = float(request.form['weight'])  
        body_temp = float(request.form['Body_Temperature'])
        heart_rate = float(request.form['Heart_Rate'])
        systolic_bp = float(request.form['Systolic_BP'])
        diastolic_bp = float(request.form['Diastolic_BP'])
        blood_glucose_hba1c = float(request.form['Blood_Glucose_HbA1c'])
        blood_glucose_fasting = float(request.form['Blood_Glucose_Fasting'])

        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)

        # Prepare input data
        user_input = {
            "Age": age,
            "Body Temperature(F) ": body_temp,
            "Heart rate(bpm)": heart_rate,
            "Systolic Blood Pressure(mm Hg)": systolic_bp,
            "Diastolic Blood Pressure(mm Hg)": diastolic_bp,
            "BMI(kg/m 2)": bmi,
            "Blood Glucose(HbA1c)": blood_glucose_hba1c,
            "Blood Glucose(Fasting hour-mg/dl)": blood_glucose_fasting
        }

        feature_order = [
            "Age", "Body Temperature(F) ", "Heart rate(bpm)", 
            "Systolic Blood Pressure(mm Hg)", "Diastolic Blood Pressure(mm Hg)",
            "BMI(kg/m 2)", "Blood Glucose(HbA1c)", "Blood Glucose(Fasting hour-mg/dl)"
        ]

        # Scale and prepare input for the model
        input_array = [[user_input[feature] for feature in feature_order]]
        input_scaled = scaler.transform(input_array)
        input_df = pd.DataFrame(input_scaled, columns=feature_order)

        # Prediction
        predicted_outcome = int(xgb_model.predict(input_df))  # Ensure integer output
        risk = risk_level(predicted_outcome)

        # Explanation
        explanations = explain_risk_factors(user_input)

        result = {
            "prediction": risk,
            "explanations": explanations
        }
        return render_template('check.html', result=result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
