
import streamlit as st
import pandas as pd
import joblib

# Load the trained machine learning pipeline
model = joblib.load("email_marketing_model.pkl")

# Configure the application
st.set_page_config(
    page_title="Email Marketing Analyzer",
    page_icon="📧",
    layout="wide"
)

# Application heading
st.title("📧 Email Marketing Campaign Effectiveness Analyzer")

st.write(
    "Analyze email campaign details and predict whether "
    "an email is likely to receive a click using machine learning."
)

st.divider()

# Sidebar
st.sidebar.title("About This Project")
st.sidebar.write(
    "This application uses an Extra Trees classification model "
    "to predict email campaign engagement."
)

st.sidebar.write(
    "Dataset: Synthetic Email Campaign Engagement Dataset"
)

# Model performance section
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "53%")
col2.metric("Precision", "50%")
col3.metric("Recall", "46.8%")
col4.metric("F1 Score", "0.484")

st.caption(
    "These are the evaluation results obtained on the test dataset."
)

st.divider()

# Input section
st.subheader("📩 Enter Campaign Details")

campaign_title = st.text_input(
    "Campaign Title",
    placeholder="Example: Summer Sale Campaign"
)

audience_name = st.text_input(
    "Audience Name",
    placeholder="Example: Regular Customers"
)

col1, col2 = st.columns(2)

with col1:
    send_date = st.date_input("Sending Date")

with col2:
    send_time = st.time_input("Sending Time")

# Prediction
if st.button("Predict Campaign Performance", type="primary"):

    if not campaign_title.strip() or not audience_name.strip():

        st.warning("Please enter both campaign title and audience name.")

    else:

        # Prepare input using the same feature names
        # that were used during model training
        input_data = pd.DataFrame([{
            "title": campaign_title,
            "name": audience_name,
            "send_month": send_date.month,
            "send_day": send_date.day,
            "send_hour": send_time.hour
        }])

        # Generate prediction
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("🎯 Prediction Result")

        # Display prediction
        if prediction == 1:
            st.success("The email is predicted to receive a click.")
        else:
            st.warning("The email is predicted not to receive a click.")

        # Display probability, if supported
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            if 1 in classes:

                click_probability = probabilities[classes.index(1)]

                st.metric(
                    "Estimated Click Probability",
                    f"{click_probability * 100:.2f}%"
                )

                st.progress(float(click_probability))

        st.caption(
            "Predictions are estimates based on patterns in the "
            "training data. Actual campaign performance may differ."
        )
