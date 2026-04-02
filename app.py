import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─── PAGE CONFIG ────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Care Assistant",
    page_icon="🏥",
    layout="wide"
)

# ─── LOAD MODEL ─────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('model/diabetes_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    with open('model/feature_names.json') as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, features = load_model()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── HELPERS ────────────────────────────────────────────
def predict_diabetes(inputs):
    df      = pd.DataFrame([inputs], columns=features)
    scaled  = scaler.transform(df)
    prob    = model.predict_proba(scaled)[0][1]
    pred    = model.predict(scaled)[0]
    return pred, prob

def get_risk_level(prob):
    if prob < 0.3:
        return "Low Risk", "green"
    elif prob < 0.6:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"

def call_groq(system, user):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# ─── HEADER ─────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#6366f1'>
🏥 Diabetes Care Assistant
</h1>
<p style='text-align:center; color:gray'>
AI-powered diabetes risk assessment + personalized health guidance
</p>
""", unsafe_allow_html=True)

st.divider()

# ─── TABS ───────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Risk Assessment",
    "💬 Health Assistant",
    "🥗 Diet Plan"
])

# ════════════════════════════════════════════════════════
# TAB 1 — RISK ASSESSMENT
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔍 Diabetes Risk Assessment")
    st.write("Enter your health metrics to assess your diabetes risk.")

    col1, col2 = st.columns(2)

    with col1:
        glucose    = st.number_input("Glucose Level (mg/dL)",
                                      min_value=50, max_value=300,
                                      value=117,
                                      help="Normal: 70-100 mg/dL fasting")
        bmi        = st.number_input("BMI",
                                      min_value=10.0, max_value=70.0,
                                      value=32.0,
                                      help="Normal: 18.5-24.9")
        age        = st.number_input("Age",
                                      min_value=18, max_value=100,
                                      value=33)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)",
                                          min_value=40, max_value=150,
                                          value=72,
                                          help="Normal: 60-80 mm Hg diastolic")

    with col2:
        pregnancies = st.number_input("Number of Pregnancies",
                                       min_value=0, max_value=20,
                                       value=1)
        insulin     = st.number_input("Insulin Level (μU/mL)",
                                       min_value=0, max_value=900,
                                       value=30,
                                       help="Normal fasting: 2-25 μU/mL")
        skin_thickness = st.number_input("Skin Thickness (mm)",
                                          min_value=0, max_value=100,
                                          value=23,
                                          help="Triceps skin fold thickness")
        dpf = st.number_input("Diabetes Pedigree Function",
                               min_value=0.0, max_value=3.0,
                               value=0.47,
                               help="Family history score (0=no history, 2.5=strong history)")

    st.divider()

    if st.button("🔍 Assess My Risk", type="primary"):
        inputs = {
            'Pregnancies':              pregnancies,
            'Glucose':                  glucose,
            'BloodPressure':            blood_pressure,
            'SkinThickness':            skin_thickness,
            'Insulin':                  insulin,
            'BMI':                      bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age':                      age
        }

        pred, prob = predict_diabetes(inputs)
        risk_level, color = get_risk_level(prob)

        # Store in session state for other tabs
        st.session_state.user_profile = inputs
        st.session_state.risk_prob    = prob
        st.session_state.risk_level   = risk_level
        st.session_state.prediction   = pred

        # Results
        st.divider()
        st.subheader("📊 Your Results")

        r1, r2, r3 = st.columns(3)
        r1.metric("Risk Level",      risk_level)
        r2.metric("Risk Probability", f"{prob*100:.1f}%")
        r3.metric("Prediction",      "Diabetic" if pred == 1 else "Non-Diabetic")

        # Risk bar
        st.progress(float(prob), text=f"Risk: {prob*100:.1f}%")

        # Key factors
        st.subheader("🔑 Your Key Risk Factors")
        importances = model.feature_importances_
        feat_imp    = sorted(zip(features, importances, inputs.values()),
                             key=lambda x: x[1], reverse=True)

        for feat, imp, val in feat_imp[:4]:
            st.write(f"**{feat}**: {val} (importance: {imp:.2f})")

        # AI recommendation
        st.subheader("💡 AI Recommendation")
        with st.spinner("Generating recommendation..."):
            rec = call_groq(
                system="You are a helpful diabetes health assistant. Give brief, practical advice.",
                user=f"""Patient profile:
Glucose: {glucose}, BMI: {bmi}, Age: {age},
Blood Pressure: {blood_pressure}, Insulin: {insulin}
Risk Level: {risk_level} ({prob*100:.1f}% probability)

Give 3 specific, actionable recommendations for this patient.
Keep it friendly and encouraging. Max 150 words."""
            )
        st.info(rec)

        st.success("✅ Assessment complete! Check the Diet Plan tab for personalized recommendations.")

# ════════════════════════════════════════════════════════
# TAB 2 — HEALTH ASSISTANT
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("💬 Diabetes Health Assistant")
    st.caption("Ask anything about diabetes — symptoms, diet, medication, lifestyle")

    # Initialize chat
    if "health_messages" not in st.session_state:
        st.session_state.health_messages = []

    # System context
    user_context = ""
    if "user_profile" in st.session_state:
        p = st.session_state.user_profile
        user_context = f"""
The user has been assessed with these metrics:
Glucose: {p['Glucose']}, BMI: {p['BMI']}, Age: {p['Age']}
Risk Level: {st.session_state.risk_level}
({st.session_state.risk_prob*100:.1f}% diabetes probability)
Personalize your answers based on their profile."""

    system = f"""You are a knowledgeable diabetes health assistant.
You help patients understand diabetes, manage their condition,
and make healthier lifestyle choices.
Always recommend consulting a doctor for medical decisions.
Be empathetic, clear, and practical.
{user_context}"""

    if st.sidebar.button("🗑️ Clear Chat", key="clear_health"):
        st.session_state.health_messages = []
        st.rerun()

    # Welcome
    if not st.session_state.health_messages:
        with st.chat_message("assistant"):
            msg = "Hi! I'm your diabetes health assistant. "
            if "risk_level" in st.session_state:
                msg += f"I can see your risk level is **{st.session_state.risk_level}**. "
            msg += "Ask me anything about diabetes, diet, symptoms, or lifestyle!"
            st.write(msg)

    # History
    for msg in st.session_state.health_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if question := st.chat_input("Ask about diabetes..."):
        with st.chat_message("user"):
            st.write(question)
        st.session_state.health_messages.append({
            "role": "user", "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = call_groq(system, question)
            st.write(answer)

        st.session_state.health_messages.append({
            "role": "assistant", "content": answer
        })

# ════════════════════════════════════════════════════════
# TAB 3 — DIET PLAN
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("🥗 Personalized Diet Plan")

    if "user_profile" not in st.session_state:
        st.warning("⚠️ Please complete the Risk Assessment first to get a personalized diet plan.")
        st.info("Go to the 🔍 Risk Assessment tab and click 'Assess My Risk'")
    else:
        p          = st.session_state.user_profile
        risk_level = st.session_state.risk_level
        prob       = st.session_state.risk_prob

        st.success(f"Generating plan for: **{risk_level}** ({prob*100:.1f}% risk)")

        # Diet preferences
        col1, col2 = st.columns(2)
        with col1:
            diet_type  = st.selectbox("Diet Preference",
                                       ["Vegetarian", "Non-Vegetarian", "Vegan"])
            cuisine    = st.selectbox("Cuisine", ["Indian", "Mediterranean",
                                                   "Asian", "Western"])
        with col2:
            allergies  = st.multiselect("Allergies/Restrictions",
                                         ["Gluten", "Dairy", "Nuts",
                                          "Soy", "Eggs", "None"])
            meals_per_day = st.selectbox("Meals per day", [3, 4, 5, 6])

        if st.button("🥗 Generate My Diet Plan", type="primary"):
            with st.spinner("Creating your personalized diet plan..."):
                plan = call_groq(
                    system="""You are a certified diabetes nutritionist.
Create detailed, practical meal plans for diabetes patients.
Focus on low glycemic index foods, portion control, and balanced nutrition.
Always include specific Indian/regional food options when relevant.""",
                    user=f"""Create a personalized 7-day meal plan for:

Patient Profile:
- Glucose: {p['Glucose']} mg/dL
- BMI: {p['BMI']}
- Age: {p['Age']}
- Risk Level: {risk_level} ({prob*100:.1f}% probability)

Preferences:
- Diet type: {diet_type}
- Cuisine: {cuisine}
- Allergies: {', '.join(allergies) if allergies else 'None'}
- Meals per day: {meals_per_day}

Include:
1. 7-day meal plan with breakfast, lunch, dinner, snacks
2. Foods to avoid
3. Foods to eat more of
4. Portion size tips
5. One diabetes-friendly recipe

Format clearly with days and meals."""
                )

            st.subheader("📋 Your 7-Day Diet Plan")
            st.write(plan)

            # Download
            st.download_button(
                label="⬇️ Download Diet Plan",
                data=plan,
                file_name=f"diabetes_diet_plan_{risk_level.lower().replace(' ','_')}.txt",
                mime="text/plain"
            )