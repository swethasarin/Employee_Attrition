import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# ===============================
# Load model + preprocessing
# ===============================
model = pickle.load(open("attrition_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

# Load dataset (for EDA)
df_original = pd.read_csv("employee_attrition.csv")

# ===============================
# Page Config + Styling
# ===============================
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")

# Background image with opacity
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)), 
                url("https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ===============================
# Sidebar Navigation
# ===============================
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["📊 Dashboard", "🔮 Prediction"])

# ===============================
# Dashboard Section
# ===============================
if section == "📊 Dashboard":
    st.title("📊 Employee Attrition Dashboard")
    st.markdown("Explore attrition patterns and employee demographics.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Attrition", data=df_original, palette="Set2", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Attrition by Department")
        fig, ax = plt.subplots()
        sns.countplot(x="Department", hue="Attrition", data=df_original, palette="Spectral", ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Monthly Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df_original["MonthlyIncome"], kde=True, color="skyblue", ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("Attrition by Job Role")
        fig, ax = plt.subplots()
        sns.countplot(y="JobRole", hue="Attrition", data=df_original, palette="coolwarm", ax=ax)
        st.pyplot(fig)

# ===============================
# Prediction Section
# ===============================
elif section == "🔮 Prediction":
    st.title("🔮 Employee Attrition Prediction")
    st.markdown("Fill in employee details and click **Predict**.")

    # Split layout
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60, 30)
        dailyrate = st.slider("Daily Rate", 100, 1500, 500)
        distancefromhome = st.slider("Distance From Home", 1, 50, 5)
        monthlyincome = st.slider("Monthly Income", 1000, 20000, 5000)

    with col2:
        totalworkingyears = st.slider("Total Working Years", 0, 40, 10)
        yearsatcompany = st.slider("Years at Company", 0, 40, 5)
        overtime = st.selectbox("OverTime", ["Yes", "No"])
        department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        jobrole = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative",
            "Manager", "Sales Representative", "Human Resources", "Other"
        ])
        businesstravel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

    # Build input dataframe
    input_df = pd.DataFrame([{
        "age": age,
        "dailyrate": dailyrate,
        "distancefromhome": distancefromhome,
        "monthlyincome": monthlyincome,
        "totalworkingyears": totalworkingyears,
        "yearsatcompany": yearsatcompany,
        "overtime": overtime.lower(),
        "department": department.lower(),
        "jobrole": jobrole.lower().replace(" ", "_"),
        "businesstravel": businesstravel.lower().replace(" ", "_"),
    }])

    if st.button("🚀 Predict"):
        with st.spinner("Analyzing employee data..."):
            time.sleep(1.5)  # simulate loading animation

            # Preprocessing
            input_df["overtime"] = input_df["overtime"].map({"yes": 1, "no": 0})
            input_encoded = pd.get_dummies(input_df, columns=["department", "jobrole", "businesstravel"], drop_first=True)

            for col in model_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_columns]

            cont_cols = scaler.feature_names_in_
            input_encoded[cont_cols] = scaler.transform(input_encoded[cont_cols])

            # Prediction
            prediction = model.predict(input_encoded)[0]
            probability = model.predict_proba(input_encoded)[0][1]

            # Result
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"🚨 The employee is **likely to leave**. (Probability: {probability:.2f})")
            else:
                st.success(f"✅ The employee is **likely to stay**. (Probability: {1-probability:.2f})")

            # Animated probability bar
            st.subheader("Attrition Probability")
            progress = st.progress(0)
            for i in range(int(probability*100)):
                time.sleep(0.01)
                progress.progress(i+1)

            st.markdown(f"**Attrition Risk: {probability*100:.1f}%**")

            # Show input data
            st.subheader("Employee Details Entered")
            st.write(input_df)