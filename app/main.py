from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('ensemble_model2.joblib')

# Dropdown options based on prior discussions
COUNTRIES = ["Kenya", "Rwanda", "Tanzania", "Uganda"]
LOCATION_TYPES = ["Urban", "Rural"]
CELLPHONE_ACCESS = ["Yes", "No"]
HOUSEHOLD_SIZE = ["1-2", "3-4", "5-6", "7+"]
GENDERS = ["Male", "Female"]
JOB_TYPES = ["Farmer", "Self-employed", "Formally employed", "Other"]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Retrieve form data
        form_data = request.form
        country = form_data["country"]
        year = int(form_data["year"])
        location_type = form_data["location_type"]
        cellphone_access = form_data["cellphone_access"] == "Yes"
        household_size = int(form_data["household_size"])
        age_of_respondent = int(form_data["age_of_respondent"])
        gender_of_respondent = form_data["gender_of_respondent"]
        relationship_with_head = form_data["relationship_with_head"]
        marital_status = form_data["marital_status"]
        education_level = form_data["education_level"]
        job_type = form_data["job_type"]

        # Derived features
        geographical_location = f"{country}_{location_type}"
        age_gender = f"{age_of_respondent}_{gender_of_respondent}"
        job_education = f"{job_type}_{education_level}"

        # Create a DataFrame for the input
        input_data = pd.DataFrame({
            "country": [country],
            "year": [year],
            "location_type": [location_type],
            "cellphone_access": [cellphone_access],
            "household_size": [household_size],
            "age_of_respondent": [age_of_respondent],
            "gender_of_respondent": [gender_of_respondent],
            "relationship_with_head": [relationship_with_head],
            "marital_status": [marital_status],
            "education_level": [education_level],
            "job_type": [job_type],
            "geographical_location": [geographical_location],
            "age_gender": [age_gender],
            "job_education": [job_education],
        })

        # One-hot encoding and preprocessing
        test_processed = pd.get_dummies(input_data, drop_first=True)
        missing_cols = set(model.feature_names_in_) - set(test_processed.columns)
        for col in missing_cols:
            test_processed[col] = 0
        test_processed = test_processed[model.feature_names_in_]

        # Predict using the model
        prediction = model.predict(test_processed)[0]
        result = "Has Bank Account" if prediction == 1 else "No Bank Account"

        return render_template("index.html", prediction=result, countries=COUNTRIES, location_types=LOCATION_TYPES,
                               cellphone_access=CELLPHONE_ACCESS, genders=GENDERS, job_types=JOB_TYPES)

    return render_template("index.html", countries=COUNTRIES, location_types=LOCATION_TYPES,
                           genders=GENDERS, job_types=JOB_TYPES)

if __name__ == "__main__":
    app.run(debug=True)
