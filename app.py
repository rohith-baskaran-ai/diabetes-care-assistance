import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import requests
from groq import Groq
from dotenv import load_dotenv

import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date 

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
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Risk Assessment",
    "💬 Health Assistant",
    "🥗 Diet Plan",
    "📊 Health Tracker"
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

# ════════════════════════════════════════════════════════
# TAB 4 — HEALTH TRACKER
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Health Tracker")
    st.caption("Track your blood sugar, BMI, exercise and medications")

    tracker_tab1, tracker_tab2, tracker_tab3, tracker_tab4 = st.tabs([
        "🩸 Blood Sugar",
        "⚖️ BMI Calculator",
        "🏃 Exercise Plan",
        "💊 Medication Notes"
    ])

    # ════════════════════════════════════════════════════
    # BLOOD SUGAR TRACKER
    # ════════════════════════════════════════════════════
    with tracker_tab1:
        st.subheader("🩸 Blood Sugar Tracker")

        # Initialize blood sugar log
        if "blood_sugar_log" not in st.session_state:
            st.session_state.blood_sugar_log = []

        # Input form
        col1, col2, col3 = st.columns(3)
        with col1:
            bs_date  = st.date_input("Date", value=date.today())
        with col2:
            bs_time  = st.selectbox("When", [
                "Fasting (morning)",
                "Before breakfast",
                "After breakfast",
                "Before lunch",
                "After lunch",
                "Before dinner",
                "After dinner",
                "Bedtime"
            ])
        with col3:
            bs_value = st.number_input(
                "Blood Sugar (mg/dL)",
                min_value=50, max_value=600, value=120
            )

        col1, col2 = st.columns(2)
        with col1:
            bs_notes = st.text_input("Notes (optional)",
                                      placeholder="e.g. after exercise, skipped meal...")
        with col2:
            st.write("")
            st.write("")
            if st.button("➕ Log Reading", type="primary"):
                st.session_state.blood_sugar_log.append({
                    "date":  str(bs_date),
                    "time":  bs_time,
                    "value": bs_value,
                    "notes": bs_notes
                })
                st.success(f"✅ Logged: {bs_value} mg/dL")

        # Blood sugar status
        def get_bs_status(value):
            if value < 70:
                return "⚠️ Low (Hypoglycemia)", "red"
            elif value <= 100:
                return "✅ Normal (Fasting)", "green"
            elif value <= 125:
                return "⚠️ Pre-diabetic range", "orange"
            elif value <= 180:
                return "⚠️ High (Post-meal)", "orange"
            else:
                return "🚨 Very High", "red"

        status, color = get_bs_status(bs_value)
        st.markdown(f"**Current reading status:** :{color}[{status}]")

        # Reference ranges
        with st.expander("📋 Blood Sugar Reference Ranges"):
            ref_data = {
                "Condition": ["Normal fasting", "Pre-diabetes fasting",
                               "Diabetes fasting", "Normal post-meal",
                               "High post-meal"],
                "Range (mg/dL)": ["70-100", "100-125", "126+",
                                   "Less than 140", "140+"]
            }
            st.table(pd.DataFrame(ref_data))

        st.divider()

        # Show log + chart
        if st.session_state.blood_sugar_log:
            st.subheader(f"📈 Your Readings ({len(st.session_state.blood_sugar_log)} logged)")

            df_log = pd.DataFrame(st.session_state.blood_sugar_log)

            # Chart
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=list(range(len(df_log))),
                y=df_log['value'],
                mode='lines+markers',
                name='Blood Sugar',
                line=dict(color='#6366f1', width=2),
                marker=dict(size=8)
            ))

            # Reference lines
            fig.add_hline(y=100, line_dash="dash",
                          line_color="green",
                          annotation_text="Normal max (100)")
            fig.add_hline(y=126, line_dash="dash",
                          line_color="red",
                          annotation_text="Diabetes threshold (126)")
            fig.add_hline(y=180, line_dash="dash",
                          line_color="orange",
                          annotation_text="Post-meal high (180)")

            fig.update_layout(
                title="Blood Sugar Trend",
                xaxis_title="Reading #",
                yaxis_title="Blood Sugar (mg/dL)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Stats
            values = df_log['value'].tolist()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Average",  f"{sum(values)/len(values):.0f} mg/dL")
            s2.metric("Highest",  f"{max(values)} mg/dL")
            s3.metric("Lowest",   f"{min(values)} mg/dL")
            s4.metric("Readings", len(values))

            # Log table
            st.subheader("📋 Reading History")
            st.dataframe(df_log, use_container_width=True)

            if st.button("🗑️ Clear All Readings"):
                st.session_state.blood_sugar_log = []
                st.rerun()
        else:
            st.info("No readings logged yet. Add your first reading above!")

    # ════════════════════════════════════════════════════
    # BMI CALCULATOR
    # ════════════════════════════════════════════════════
    with tracker_tab2:
        st.subheader("⚖️ BMI Calculator")

        col1, col2 = st.columns(2)
        with col1:
            height_cm = st.number_input("Height (cm)",
                                         min_value=100, max_value=250,
                                         value=170)
            weight_kg = st.number_input("Weight (kg)",
                                         min_value=30, max_value=300,
                                         value=70)

            bmi_val = weight_kg / ((height_cm/100) ** 2)

            def get_bmi_category(bmi):
                if bmi < 18.5:
                    return "Underweight", "#3b82f6"
                elif bmi < 25:
                    return "Normal", "#22c55e"
                elif bmi < 30:
                    return "Overweight", "#f59e0b"
                else:
                    return "Obese", "#ef4444"

            category, cat_color = get_bmi_category(bmi_val)

            st.metric("Your BMI", f"{bmi_val:.1f}")
            st.markdown(f"**Category:** <span style='color:{cat_color}'>{category}</span>",
                        unsafe_allow_html=True)

            # Ideal weight range
            ideal_min = 18.5 * ((height_cm/100) ** 2)
            ideal_max = 24.9 * ((height_cm/100) ** 2)
            st.info(f"Ideal weight for your height: **{ideal_min:.1f} - {ideal_max:.1f} kg**")

        with col2:
            # BMI Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bmi_val,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "BMI", 'font': {'color': 'white'}},
                gauge={
                    'axis': {'range': [10, 45],
                             'tickcolor': 'white',
                             'tickfont': {'color': 'white'}},
                    'bar':  {'color': cat_color},
                    'steps': [
                        {'range': [10, 18.5], 'color': '#3b82f6'},
                        {'range': [18.5, 25], 'color': '#22c55e'},
                        {'range': [25, 30],   'color': '#f59e0b'},
                        {'range': [30, 45],   'color': '#ef4444'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': bmi_val
                    }
                }
            ))

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        # BMI table
        with st.expander("📋 BMI Categories"):
            bmi_table = {
                "Category":    ["Underweight", "Normal", "Overweight", "Obese"],
                "BMI Range":   ["Below 18.5", "18.5 - 24.9",
                                "25.0 - 29.9", "30.0 and above"],
                "Diabetes Risk": ["Low", "Low", "Moderate", "High"]
            }
            st.table(pd.DataFrame(bmi_table))

    # ════════════════════════════════════════════════════
    # EXERCISE PLAN
    # ════════════════════════════════════════════════════
    with tracker_tab3:
        st.subheader("🏃 Exercise Recommendations")

        fitness_level = st.selectbox("Your current fitness level",
                                      ["Sedentary (little exercise)",
                                       "Lightly active (1-3 days/week)",
                                       "Moderately active (3-5 days/week)",
                                       "Very active (6-7 days/week)"])

        exercise_goal = st.selectbox("Your goal",
                                      ["Lower blood sugar",
                                       "Lose weight",
                                       "Improve cardiovascular health",
                                       "Increase energy levels"])

        limitations = st.multiselect("Any physical limitations?",
                                      ["Joint pain", "Back pain",
                                       "Heart condition", "None"])

        if st.button("🏃 Generate Exercise Plan", type="primary"):
            risk = st.session_state.get('risk_level', 'Medium Risk')
            prob = st.session_state.get('risk_prob', 0.5)

            with st.spinner("Creating your exercise plan..."):
                plan = call_groq(
                    system="""You are a certified diabetes fitness coach.
Create safe, practical exercise plans for diabetes patients.
Always prioritize safety and recommend consulting a doctor.""",
                    user=f"""Create a weekly exercise plan for:
- Diabetes Risk: {risk} ({prob*100:.0f}%)
- Fitness Level: {fitness_level}
- Goal: {exercise_goal}
- Limitations: {', '.join(limitations) if limitations else 'None'}

Include:
1. Weekly schedule (7 days)
2. Specific exercises with duration
3. Warm-up and cool-down
4. How each exercise helps blood sugar
5. Safety tips for diabetic patients

Keep it practical and motivating."""
                )

            st.subheader("🏋️ Your Weekly Exercise Plan")
            st.write(plan)

            st.download_button(
                "⬇️ Download Exercise Plan",
                data=plan,
                file_name="exercise_plan.txt",
                mime="text/plain"
            )

    # ════════════════════════════════════════════════════
    # MEDICATION NOTES
    # ════════════════════════════════════════════════════
    with tracker_tab4:
        st.subheader("💊 Medication Notes")
        st.caption("Keep track of your medications — always consult your doctor")

        # Initialize
        if "medications" not in st.session_state:
            st.session_state.medications = []

        # Add medication
        st.subheader("➕ Add Medication")
        col1, col2, col3 = st.columns(3)
        with col1:
            med_name = st.text_input("Medication Name",
                                      placeholder="e.g. Metformin")
        with col2:
            med_dose = st.text_input("Dosage",
                                      placeholder="e.g. 500mg")
        with col3:
            med_freq = st.selectbox("Frequency", [
                "Once daily",
                "Twice daily",
                "Three times daily",
                "With meals",
                "Before meals",
                "After meals",
                "At bedtime"
            ])

        med_notes = st.text_input("Additional notes",
                                   placeholder="e.g. take with food, avoid alcohol")

        if st.button("➕ Add Medication", type="primary"):
            if med_name:
                st.session_state.medications.append({
                    "name":      med_name,
                    "dose":      med_dose,
                    "frequency": med_freq,
                    "notes":     med_notes
                })
                st.success(f"✅ Added: {med_name}")
            else:
                st.error("Please enter medication name")

        # Show medications
        if st.session_state.medications:
            st.divider()
            st.subheader("💊 Your Medications")

            for i, med in enumerate(st.session_state.medications):
                with st.expander(f"💊 {med['name']} — {med['dose']} — {med['frequency']}"):
                    st.write(f"**Name:** {med['name']}")
                    st.write(f"**Dose:** {med['dose']}")
                    st.write(f"**Frequency:** {med['frequency']}")
                    if med['notes']:
                        st.write(f"**Notes:** {med['notes']}")
                    if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.medications.pop(i)
                        st.rerun()

            # Download medication list
            med_text = "My Medications\n" + "="*30 + "\n\n"
            for med in st.session_state.medications:
                med_text += f"• {med['name']} — {med['dose']} — {med['frequency']}\n"
                if med['notes']:
                    med_text += f"  Notes: {med['notes']}\n"
                med_text += "\n"
            med_text += "\n⚠️ Always consult your doctor before changing medications."

            st.download_button(
                "⬇️ Download Medication List",
                data=med_text,
                file_name="my_medications.txt",
                mime="text/plain"
            )
        else:
            st.info("No medications added yet.")

        st.divider()
        st.warning("⚠️ This is for personal tracking only. Always follow your doctor's prescription.")
