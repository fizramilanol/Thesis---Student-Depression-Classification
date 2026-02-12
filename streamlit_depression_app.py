import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Depression Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model components
@st.cache_resource
def load_model_components():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'depression_prediction_model.joblib')
    try:
        components = joblib.load('depression_prediction_model.joblib')
        return components
    except FileNotFoundError:
        st.error("Model file 'depression_prediction_model.joblib' not found! Please ensure it's in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model_components = load_model_components()
model = model_components['model']
feature_names = model_components['feature_names']
encoding_maps = model_components['encoding_maps']
target_map = model_components['target_map']

# Invert the target map for displaying results
inverted_target_map = {v: k for k, v in target_map.items()}

# Extract mapping dictionaries directly from the loaded components
Gender_map = encoding_maps['Gender']
City_map = encoding_maps['City']
Sleep_Duration_map = encoding_maps['Sleep_Duration']
Dietary_Habits_map = encoding_maps['Dietary_Habits']
Degree_map = encoding_maps['Degree']
Suicidal_Thoughts_map = encoding_maps['Suicidal_Thoughts']
Family_History_of_Mental_Illness_map = encoding_maps['Family_History_of_Mental_Illness']

# Invert the mapping dictionaries for Streamlit selectbox options
inverted_gender_map = {v: k for k, v in Gender_map.items()}
inverted_city_map = {v: k for v, k in City_map.items()}
inverted_sleep_duration_map = {v: k for k, v in Sleep_Duration_map.items()}
inverted_dietary_habits_map = {v: k for k, v in Dietary_Habits_map.items()}
inverted_degree_map = {v: k for k, v in Degree_map.items()}
inverted_suicidal_thoughts_map = {v: k for k, v in Suicidal_Thoughts_map.items()}
inverted_family_history_map = {v: k for k, v in Family_History_of_Mental_Illness_map.items()}

def predict_depression(data, model_components):
    df = pd.DataFrame([data])

    # Apply encoding maps to the input data
    df['Gender'] = df['Gender'].map(Gender_map)
    df['City'] = df['City'].map(City_map)
    df['Sleep_Duration'] = df['Sleep_Duration'].map(Sleep_Duration_map)
    df['Dietary_Habits'] = df['Dietary_Habits'].map(Dietary_Habits_map)
    df['Degree'] = df['Degree'].map(Degree_map)
    df['Suicidal_Thoughts'] = df['Suicidal_Thoughts'].map(Suicidal_Thoughts_map)
    df['Family_History_of_Mental_Illness'] = df['Family_History_of_Mental_Illness'].map(Family_History_of_Mental_Illness_map)

    model = model_components['model']
    feature_names = model_components['feature_names']
    df_for_pred = df[feature_names].copy()

    prediction = model.predict(df_for_pred)[0]
    probabilities = model.predict_proba(df_for_pred)[0]

    return {
        'prediction': int(prediction),
        'prediction_label': inverted_target_map[prediction], # Use inverted_target_map
        'probability': float(probabilities[prediction]), # Probability for the predicted class
        'probabilities': probabilities.tolist()
    }

def export_prediction(data, result):
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'input_data': data,
        'prediction': {
            'class': result['prediction_label'],
            'confidence': result['probability'],
            'raw_prediction': result['prediction']
        }
    }
    return json.dumps(export_data, indent=2)

def reset_session_state():
    for key in ['Gender', 'Age', 'City', 'Academic_Pressure', 'CGPA', 'Study_Satisfaction', 'Sleep_Duration', 'Dietary_Habits', 'Degree', 'Suicidal_Thoughts', 'Work/Study_Hours', 'Financial_Stress', 'Family_History_of_Mental_Illness']:
        if key in st.session_state:
            del st.session_state[key]

# Define options for Streamlit input widgets
gender_options = list(inverted_gender_map.values())
city_options = list(inverted_city_map.values())
sleep_duration_options = list(inverted_sleep_duration_map.values())
dietary_habits_options = list(inverted_dietary_habits_map.values())
degree_options = list(inverted_degree_map.values())
suicidal_thoughts_options = list(inverted_suicidal_thoughts_map.values())
family_history_options = list(inverted_family_history_map.values())

# App title
st.title("🧠 Depression Prediction App")
st.markdown("Predict the likelihood of depression based on various personal and academic factors.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Input Your Information")
    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            gender = st.selectbox("Gender", gender_options, key="Gender")
            age = st.slider("Age", 18, 60, 25, key="Age")
            academic_pressure = st.slider("Academic Pressure (1-5)", 0.0, 5.0, 3.0, key="Academic_Pressure")
            study_satisfaction = st.slider("Study Satisfaction (1-5)", 0.0, 5.0, 3.0, key="Study_Satisfaction")

        with col_b:
            city = st.selectbox("City", city_options, key="City")
            cgpa = st.number_input("CGPA (0.0 - 10.0)", min_value=0.0, max_value=10.0, value=7.5, step=0.1, key="CGPA")
            work_study_hours = st.slider("Work/Study Hours (0-12)", 0.0, 12.0, 8.0, key="Work/Study_Hours")
            dietary_habits = st.selectbox("Dietary Habits", dietary_habits_options, key="Dietary_Habits")

        with col_c:
            degree = st.selectbox("Degree", degree_options, key="Degree")
            sleep_duration = st.selectbox("Sleep Duration", sleep_duration_options, key="Sleep_Duration")
            financial_stress = st.slider("Financial Stress (1-5)", 1.0, 5.0, 3.0, key="Financial_Stress")
            family_history_mi = st.selectbox("Family History of Mental Illness", family_history_options, key="Family_History_of_Mental_Illness")
            suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", suicidal_thoughts_options, key="Suicidal_Thoughts")

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            predict_button = st.form_submit_button("🔮 Predict", type="primary")
        with col_btn2:
            reset_button = st.form_submit_button("🔄 Reset")
        with col_btn3:
            export_button = st.form_submit_button("📤 Export Last Result")

# Reset handler
if reset_button:
    reset_session_state()
    st.rerun()

# Predict handler
if predict_button:
    input_data = {
        'Gender': gender,
        'Age': float(age),
        'City': city,
        'Academic_Pressure': float(academic_pressure),
        'CGPA': float(cgpa),
        'Study_Satisfaction': float(study_satisfaction),
        'Sleep_Duration': sleep_duration,
        'Dietary_Habits': dietary_habits,
        'Degree': degree,
        'Suicidal_Thoughts': suicidal_thoughts,
        'Work/Study_Hours': float(work_study_hours),
        'Financial_Stress': float(financial_stress),
        'Family_History_of_Mental_Illness': family_history_mi
    }

    try:
        result = predict_depression(input_data, model_components)
        st.session_state['last_prediction'] = {'input_data': input_data, 'result': result}
        
        with col2:
            st.subheader("🎯 Prediction Results")
            st.markdown(f"**Predicted Class:** `{result['prediction_label']}`")
    
            # Gauge
            confidence = result['probability'] * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Confidence (%)"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Probability chart
            prob_df = pd.DataFrame({
                'Class': list(class_map.keys()),
                'Probability': result['probabilities']
            })

            fig_bar = px.bar(prob_df, x='Class', y='Probability', color='Probability',
                            color_continuous_scale='viridis',
                            title='Class Probability Distribution')
            fig_bar.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")

# Feature importance
st.subheader("📊 Feature Importance")
if 'model' in model_components:
    try:
        importance_df = pd.DataFrame({
            'Feature': model_components['feature_names'],
            'Importance': model_components['model'].feature_importances_
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                         title='Feature Importance', color='Importance',
                         color_continuous_scale='plasma')
        fig_imp.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error displaying feature importance: {str(e)}")

# Export
if export_button:
    if 'last_prediction' in st.session_state:
        export_data = export_prediction(
            st.session_state['last_prediction']['input_data'],
            st.session_state['last_prediction']['result']
        )
        st.download_button(
            label="📥 Download Prediction Results",
            data=export_data,
            file_name=f"car_evaluation_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    else:
        st.warning("⚠️ No prediction results to export. Please make a prediction first.")

# Footer
st.markdown("---")

st.markdown("*Built with Streamlit • Car Evaluation App*")
