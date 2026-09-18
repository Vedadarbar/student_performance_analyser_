import streamlit as st
import pandas as pd
import pickle


# ==============================
# Load Model and Scaler
# ==============================

with open("student_performance_svm.pkl", "rb") as file:
    model = pickle.load(file)

with open("student_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="Student Performance",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Prediction")

st.write("Enter Student Details below:")

st.markdown("---")


# ==============================
# User Inputs
# ==============================

hours_studied = st.number_input(
    "Hours Studied",
    min_value=0,
    max_value=24,
    value=5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0,
    max_value=100,
    value=75
)


# Parental Involvement
parental_involvement = st.selectbox(
    "Parental Involvement",
    [0, 1, 2],
    format_func=lambda x: {
        0: "0 - Low",
        1: "1 - Medium",
        2: "2 - High"
    }[x]
)


# Access to Resources
access_to_resources = st.selectbox(
    "Access to Resources",
    [0, 1, 2],
    format_func=lambda x: {
        0: "0 - Low",
        1: "1 - Medium",
        2: "2 - High"
    }[x]
)


# Extracurricular Activities
extracurricular_activities = st.selectbox(
    "Extracurricular Activities",
    [0, 1],
    format_func=lambda x: {
        0: "0 - No",
        1: "1 - Yes"
    }[x]
)


sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0,
    max_value=24,
    value=7
)

previous_scores = st.number_input(
    "Previous Scores",
    min_value=0,
    max_value=100,
    value=60
)


# Motivation
motivation_level = st.selectbox(
    "Motivation Level",
    [0, 1, 2],
    format_func=lambda x: {
        0: "0 - Low",
        1: "1 - Medium",
        2: "2 - High"
    }[x]
)


# Internet Access
internet_access = st.selectbox(
    "Internet Access",
    [0, 1],
    format_func=lambda x: {
        0: "0 - No",
        1: "1 - Yes"
    }[x]
)


tutoring_sessions = st.number_input(
    "Tutoring Sessions",
    min_value=0,
    max_value=20,
    value=2
)


# Family Income
family_income = st.selectbox(
    "Family Income",
    [0, 1, 2],
    format_func=lambda x: {
        0: "0 - Low",
        1: "1 - Medium",
        2: "2 - High"
    }[x]
)


# Teacher Quality
teacher_quality = st.selectbox(
    "Teacher Quality",
    [0, 1, 2],
    format_func=lambda x: {
        0: "0 - Low",
        1: "1 - Medium",
        2: "2 - High"
    }[x]
)


# School Type
school_type = st.selectbox(
    "School Type",
    [0, 1],
    format_func=lambda x: {
        0: "0 - Public",
        1: "1 - Private"
    }[x]
)


# Peer Influence
peer_influence = st.selectbox(
    "Peer Influence",
    [0, 1, 2],
    format_func=lambda x: {
        0: "0 - Negative",
        1: "1 - Neutral",
        2: "2 - Positive"
    }[x]
)


physical_activity = st.number_input(
    "Physical Activity (hours)",
    min_value=0,
    max_value=24,
    value=5
)


# Learning Disabilities
learning_disabilities = st.selectbox(
    "Learning Disabilities",
    [0, 1],
    format_func=lambda x: {
        0: "0 - No",
        1: "1 - Yes"
    }[x]
)


# Gender
gender = st.selectbox(
    "Gender",
    [0, 1],
    format_func=lambda x: {
        0: "0 - Female",
        1: "1 - Male"
    }[x]
)


# ==============================
# Prediction
# ==============================

if st.button("Predict Exam Score"):

    input_data = pd.DataFrame([[
        hours_studied,
        attendance,
        parental_involvement,
        access_to_resources,
        extracurricular_activities,
        sleep_hours,
        previous_scores,
        motivation_level,
        internet_access,
        tutoring_sessions,
        family_income,
        teacher_quality,
        school_type,
        peer_influence,
        physical_activity,
        learning_disabilities,
        gender
    ]],
    columns=[
        "Hours_Studied",
        "Attendance",
        "Parental_Involvement",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Sleep_Hours",
        "Previous_Scores",
        "Motivation_Level",
        "Internet_Access",
        "Tutoring_Sessions",
        "Family_Income",
        "Teacher_Quality",
        "School_Type",
        "Peer_Influence",
        "Physical_Activity",
        "Learning_Disabilities",
        "Gender"
    ])


    # ==============================
    # Scaling
    # ==============================

    input_scaled = scaler.transform(input_data)


    # ==============================
    # Prediction
    # ==============================

    prediction = model.predict(input_scaled)


    # ==============================
    # Display Result
    # ==============================

    st.markdown("---")

    st.success(
        f"🎯 Predicted Exam Score: {prediction[0]:.2f}"
    )