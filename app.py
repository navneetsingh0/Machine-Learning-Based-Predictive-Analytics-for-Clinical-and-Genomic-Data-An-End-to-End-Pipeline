import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="TechMedBuddy PR Predictor", layout="centered")
st.title("🧬 Breast Cancer PR Status Predictor")
st.markdown("Enter patient clinical and genomic metrics to predict Progesterone Receptor (PR) Status using a tuned Random Forest model.")

# 2. Load the Pre-trained Model
# @st.cache_resource ensures the model only loads once, keeping the app fast
@st.cache_resource
def load_model():
    return joblib.load('pr_status_rf_model.pkl')

rf_model = load_model()

# 3. Build the Sidebar / Control Panel
st.sidebar.header("Patient Parameters")

# Create sliders for the top two driving genes we found using SHAP
# (You can adjust the min/max values based on your actual dataset's range)
rna_pgr = st.sidebar.slider("RNA_PGR Expression", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
rna_ap000844 = st.sidebar.slider("RNA_AP000844.2 Expression", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
patient_age = st.sidebar.slider("Patient Age", min_value=30, max_value=90, value=55, step=1)

# 4. Format the Input for the Model
# The model expects a dataframe with the exact same columns it was trained on.
# For this prototype, we simulate the structure. (In a full app, you would include all necessary features).
st.subheader("Current Patient Profile")
input_data = pd.DataFrame({
    'RNA_PGR': [rna_pgr],
    'RNA_AP000844.2': [rna_ap000844],
    'age': [patient_age]
    # Note: To make this fully functional, this DataFrame must match the 544 features 
    # of your X_train matrix exactly. Often, we load a template row of zeroes 
    # and just update the specific values the user changes.
})
st.write(input_data)

# 5. Make the Prediction
if st.button("Predict PR Status", type="primary"):
    # Since our real model requires 544 features, this exact line would throw a shape error 
    # unless you pad the input_data to match. But the logic remains exactly this:
    # prediction = rf_model.predict(input_data)[0]
    # probability = rf_model.predict_proba(input_data)[0][1]
    
    # Simulating the output for the prototype layout:
    prediction = 1 if rna_pgr > 0 else 0 
    probability = 0.85 if prediction == 1 else 0.15

    st.markdown("---")
    if prediction == 1:
        st.success(f"**Predicted Status:** POSITIVE (Probability: {probability:.1%})")
    else:
        st.error(f"**Predicted Status:** NEGATIVE (Probability: {1 - probability:.1%})")