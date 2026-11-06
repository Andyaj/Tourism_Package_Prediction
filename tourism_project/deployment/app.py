import os
import streamlit as st
import pandas as pd
import joblib

# Load the model that the GitHub Actions pipeline committed next to this file
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

st.title("Wellness Tourism Package Purchase Prediction")
st.write(
    "Enter the customer details and the sales interaction details below. "
    "The model predicts whether the customer is likely to purchase the Wellness Tourism Package."
)

# Customer details
st.subheader("Customer details")
Age = st.slider("Age", 18, 70, 36)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Divorced", "Unmarried", "Single"])
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=22000.0, step=500.0)
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", 1, 5, 3)
NumberOfChildrenVisiting = st.slider("Number of Children Visiting (below age 5)", 0, 3, 1)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
NumberOfTrips = st.slider("Number of Trips per Year", 1, 22, 3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

# Customer interaction details
st.subheader("Sales interaction details")
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
DurationOfPitch = st.slider("Duration of Pitch (minutes)", 5, 130, 14)
NumberOfFollowups = st.slider("Number of Follow-ups", 1, 6, 4)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# Collect the inputs into a single-row dataframe with the same columns used in training
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

# Same threshold that was used to evaluate the model in train.py
classification_threshold = 0.45

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)
    st.write("Input sent to the model:")
    st.dataframe(input_data)
    st.write(f"Predicted purchase probability: {prob:.2%}")
    if pred == 1:
        st.success("Prediction: the customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.info("Prediction: the customer is unlikely to purchase the Wellness Tourism Package.")
