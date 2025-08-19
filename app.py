from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = joblib.load('heart_model.pkl')
except FileNotFoundError:
    print("Error: Model file 'heart_model.pkl' not found.")
    exit()

# Define mappings for OneHotEncoding (based on heart_2020_cleaned.csv)
mappings = {
    'Smoking': {'Yes': 'Smoking_Yes', 'No': 'Smoking_No'},
    'AlcoholDrinking': {'Yes': 'AlcoholDrinking_Yes', 'No': 'AlcoholDrinking_No'},
    'Stroke': {'Yes': 'Stroke_Yes', 'No': 'Stroke_No'},
    'DiffWalking': {'Yes': 'DiffWalking_Yes', 'No': 'DiffWalking_No'},
    'Sex': {'Male': 'Sex_Male', 'Female': 'Sex_Female'},
    'AgeCategory': {
        '18-24': 'AgeCategory_18-24', '25-29': 'AgeCategory_25-29', '30-34': 'AgeCategory_30-34',
        '35-39': 'AgeCategory_35-39', '40-44': 'AgeCategory_40-44', '45-49': 'AgeCategory_45-49',
        '50-54': 'AgeCategory_50-54', '55-59': 'AgeCategory_55-59', '60-64': 'AgeCategory_60-64',
        '65-69': 'AgeCategory_65-69', '70-74': 'AgeCategory_70-74', '75-79': 'AgeCategory_75-79',
        '80 or older': 'AgeCategory_80 or older'
    },
    'Race': {
        'White': 'Race_White', 'Black': 'Race_Black', 'Asian': 'Race_Asian',
        'American Indian/Alaskan Native': 'Race_American Indian/Alaskan Native',
        'Hispanic': 'Race_Hispanic', 'Other': 'Race_Other'
    },
    'Diabetic': {
        'Yes': 'Diabetic_Yes', 'No': 'Diabetic_No',
        'No, borderline diabetes': 'Diabetic_No, borderline diabetes',
        'Yes (during pregnancy)': 'Diabetic_Yes (during pregnancy)'
    },
    'PhysicalActivity': {'Yes': 'PhysicalActivity_Yes', 'No': 'PhysicalActivity_No'},
    'GenHealth': {
        'Excellent': 'GenHealth_Excellent', 'Very good': 'GenHealth_Very good',
        'Good': 'GenHealth_Good', 'Fair': 'GenHealth_Fair', 'Poor': 'GenHealth_Poor'
    },
    'Asthma': {'Yes': 'Asthma_Yes', 'No': 'Asthma_No'},
    'KidneyDisease': {'Yes': 'KidneyDisease_Yes', 'No': 'KidneyDisease_No'},
    'SkinCancer': {'Yes': 'SkinCancer_Yes', 'No': 'SkinCancer_No'}
}


# Route to handle both form display and prediction
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    prob_yes = None
    prob_no = None
    error = None

    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                'BMI': float(request.form['BMI']),
                'PhysicalHealth': float(request.form['PhysicalHealth']),
                'MentalHealth': float(request.form['MentalHealth']),
                'SleepTime': float(request.form['SleepTime'])
            }

            # Handle categorical variables with OneHotEncoding
            categorical_fields = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex',
                                  'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity',
                                  'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer']

            for field in categorical_fields:
                value = request.form[field]
                if field in mappings and value in mappings[field]:
                    input_data[mappings[field][value]] = 1
                else:
                    error = f"Invalid value for {field}"
                    return render_template('index.html', error=error)

            # Create DataFrame
            input_df = pd.DataFrame([input_data])

            # Ensure columns match the model's expected input
            model_columns = model.feature_names_in_
            input_df = input_df.reindex(columns=model_columns, fill_value=0)

            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            prob_no = round(probability[0] * 100, 2)
            prob_yes = round(probability[1] * 100, 2)
            prediction = 'Yes' if prediction == 1 else 'No'

        except Exception as e:
            error = str(e)

    return render_template('index.html', prediction=prediction, prob_yes=prob_yes, prob_no=prob_no, error=error)


if __name__ == '__main__':
    app.run(debug=True)