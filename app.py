import io
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

def call_groq(system, user, history=None):
    messages = [{"role": "system", "content": system}]
    
    # add last 10 messages if history exists
    if history:
        for msg in history[-10:]:
            messages.append({
                "role":    msg["role"],
                "content": msg["content"]
            })
    
    messages.append({"role": "user", "content": user})
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1000
    )
    return response.choices[0].message.content

def generate_health_report(profile, risk_data, bs_log, med_log, exercise_log, food_log):

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    import io
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    story.append(Paragraph("🏥 Diabetes Care Health Report", styles['Title']))
    story.append(Paragraph(f"Generated: {date.today().strftime('%d %B %Y')}", styles['Normal']))
    story.append(Spacer(1, 20))

    # Patient Info
    story.append(Paragraph("Patient Information", styles['Heading1']))
    if profile:
        patient_data = [
            ["Name", profile.get('name', 'N/A')],
            ["Age", str(profile.get('age', 'N/A'))],
            ["Gender", profile.get('gender', 'N/A')],
            ["Diabetes Status", profile.get('diabetes_type', 'N/A')],
            ["BMI", str(profile.get('bmi', 'N/A'))],
            ["Diet Preference", profile.get('diet_pref', 'N/A')],
        ]
        t = Table(patient_data, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.lightblue),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
    story.append(Spacer(1, 20))

    # Risk Assessment
    story.append(Paragraph("Risk Assessment", styles['Heading1']))
    if risk_data:
        story.append(Paragraph(
            f"Risk Level: {risk_data.get('level', 'N/A')} "
            f"({risk_data.get('prob', 0)*100:.1f}% probability)",
            styles['Normal']
        ))
    else:
        story.append(Paragraph("No assessment completed yet.", styles['Normal']))
    story.append(Spacer(1, 20))

    # Blood Sugar Summary
    story.append(Paragraph("Blood Sugar Summary", styles['Heading1']))
    if bs_log:
        values  = [r['value'] for r in bs_log]
        avg_bs  = sum(values) / len(values)
        # A1C estimation
        a1c_est = (avg_bs + 46.7) / 28.7
        bs_summary = [
            ["Total Readings", str(len(bs_log))],
            ["Average Blood Sugar", f"{avg_bs:.0f} mg/dL"],
            ["Highest Reading", f"{max(values)} mg/dL"],
            ["Lowest Reading", f"{min(values)} mg/dL"],
            ["Estimated A1C", f"{a1c_est:.1f}%"],
        ]
        t = Table(bs_summary, colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.lightyellow),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("No blood sugar readings logged.", styles['Normal']))
    story.append(Spacer(1, 20))

    # Medications
    story.append(Paragraph("Current Medications", styles['Heading1']))
    if med_log:
        for med in med_log:
            story.append(Paragraph(
                f"• {med['name']} — {med['dose']} — {med['frequency']}",
                styles['Normal']
            ))
    else:
        story.append(Paragraph("No medications logged.", styles['Normal']))
    story.append(Spacer(1, 20))

    # Exercise Summary
    story.append(Paragraph("Exercise Summary", styles['Heading1']))
    if exercise_log:
        total_mins = sum(r['duration'] for r in exercise_log)
        total_cals = sum(r['calories'] for r in exercise_log)
        story.append(Paragraph(
            f"Total sessions: {len(exercise_log)} | "
            f"Total minutes: {total_mins} | "
            f"Total calories: {total_cals}",
            styles['Normal']
        ))
    else:
        story.append(Paragraph("No exercise logged.", styles['Normal']))
    story.append(Spacer(1, 20))

    # Food Summary
    story.append(Paragraph("Nutrition Summary", styles['Heading1']))
    if food_log:
        total_carbs = sum(r['carbs'] for r in food_log)
        total_cals  = sum(r['calories'] for r in food_log)
        avg_gi      = sum(r['gi'] for r in food_log if r['gi'] > 0)
        story.append(Paragraph(
            f"Total meals logged: {len(food_log)} | "
            f"Total carbs: {total_carbs}g | "
            f"Total calories: {total_cals}",
            styles['Normal']
        ))
    else:
        story.append(Paragraph("No food logged.", styles['Normal']))

    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "⚠️ This report is for personal tracking only. "
        "Always consult your doctor for medical advice.",
        styles['Normal']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer
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
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 My Profile",
    "🔍 Risk Assessment",
    "💬 Health Assistant",
    "🥗 Diet Plan",
    "📊 Health Tracker",
    "📈 Dashboard"
])

# ════════════════════════════════════════════════════════
# TAB 0 — PATIENT PROFILE
# ════════════════════════════════════════════════════════
with tab0:
    st.subheader("👤 My Health Profile")
    st.caption("Fill this once — all tabs will use your profile automatically")

    # Initialize profile
    if "profile" not in st.session_state:
        st.session_state.profile = {}

    profile = st.session_state.profile

    # ── Personal Info ────────────────────────────────────
    st.markdown("### 🧑 Personal Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        name   = st.text_input("Full Name",
                                value=profile.get("name", ""),
                                placeholder="Your name")
    with col2:
        age    = st.number_input("Age",
                                  min_value=18, max_value=100,
                                  value=profile.get("age", 35))
    with col3:
        gender = st.selectbox("Gender",
                               ["Male", "Female", "Other"],
                               index=["Male", "Female", "Other"].index(
                                   profile.get("gender", "Male")))

    # ── Physical Stats ───────────────────────────────────
    st.markdown("### 📏 Physical Stats")
    col1, col2, col3 = st.columns(3)
    with col1:
        height = st.number_input("Height (cm)",
                                  min_value=100, max_value=250,
                                  value=profile.get("height", 170))
    with col2:
        weight = st.number_input("Weight (kg)",
                                  min_value=30, max_value=300,
                                  value=profile.get("weight", 70))
    with col3:
        # Auto calculate BMI
        bmi_auto = weight / ((height/100) ** 2)
        st.metric("Your BMI", f"{bmi_auto:.1f}")

    # ── Medical Info ─────────────────────────────────────
    st.markdown("### 🏥 Medical Information")
    col1, col2 = st.columns(2)
    with col1:
        diabetes_type = st.selectbox(
            "Diabetes Status",
            ["Not diagnosed", "At risk", "Pre-diabetic",
             "Type 2 Diabetes", "Type 1 Diabetes"],
            index=["Not diagnosed", "At risk", "Pre-diabetic",
                   "Type 2 Diabetes", "Type 1 Diabetes"].index(
                       profile.get("diabetes_type", "Not diagnosed"))
        )
        diet_pref = st.selectbox(
            "Diet Preference",
            ["Vegetarian", "Non-Vegetarian", "Vegan"],
            index=["Vegetarian", "Non-Vegetarian", "Vegan"].index(
                profile.get("diet_pref", "Vegetarian")), 
                key="profile_diet_type"
        )
        cuisine = st.selectbox(
            "Cuisine Preference",
            ["Indian", "Mediterranean", "Asian", "Western"],
            index=["Indian", "Mediterranean", "Asian", "Western"].index(
                profile.get("cuisine", "Indian"))
        )

    with col2:
        conditions = st.multiselect(
            "Other conditions",
            ["Hypertension", "High Cholesterol", "Obesity",
             "Heart Disease", "Kidney Disease", "None"],
            default=profile.get("conditions", ["None"])
        )
        allergies = st.multiselect(
            "Food Allergies",
            ["Gluten", "Dairy", "Nuts", "Soy", "Eggs", "None"],
            default=profile.get("allergies", ["None"])
        )
        fitness_level = st.selectbox(
            "Fitness Level",
            ["Sedentary", "Lightly active",
             "Moderately active", "Very active"],
            index=["Sedentary", "Lightly active",
                   "Moderately active", "Very active"].index(
                       profile.get("fitness_level", "Sedentary"))
        )

    # ── Blood Sugar Targets ──────────────────────────────
    st.markdown("### 🎯 Personal Targets")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_fasting = st.number_input(
            "Target Fasting Blood Sugar (mg/dL)",
            min_value=70, max_value=150,
            value=profile.get("target_fasting", 100),
            help="Normal: 70-100 mg/dL"
        )
    with col2:
        target_postmeal = st.number_input(
            "Target Post-meal Blood Sugar (mg/dL)",
            min_value=100, max_value=250,
            value=profile.get("target_postmeal", 140),
            help="Normal: less than 140 mg/dL"
        )
    with col3:
        target_weight = st.number_input(
            "Target Weight (kg)",
            min_value=30, max_value=200,
            value=profile.get("target_weight", weight)
        )

    st.divider()

    # ── Save Profile ─────────────────────────────────────
    if st.button("💾 Save Profile", type="primary"):
        st.session_state.profile = {
            "name":            name,
            "age":             age,
            "gender":          gender,
            "height":          height,
            "weight":          weight,
            "bmi":             round(bmi_auto, 1),
            "diabetes_type":   diabetes_type,
            "diet_pref":       diet_pref,
            "cuisine":         cuisine,
            "conditions":      conditions,
            "allergies":       allergies,
            "fitness_level":   fitness_level,
            "target_fasting":  target_fasting,
            "target_postmeal": target_postmeal,
            "target_weight":   target_weight
        }
        st.success(f"✅ Profile saved! Welcome, {name}!")
        st.balloons()

    # ── Profile Summary ──────────────────────────────────
    if st.session_state.profile:
        st.divider()
        p = st.session_state.profile
        st.markdown("### 📋 Your Profile Summary")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Name",         p.get('name', '-'))
        col2.metric("Age",          p.get('age', '-'))
        col3.metric("BMI",          p.get('bmi', '-'))
        col4.metric("Status",       p.get('diabetes_type', '-'))

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Diet",         p.get('diet_pref', '-'))
        col2.metric("Cuisine",      p.get('cuisine', '-'))
        col3.metric("Target Fast",  f"{p.get('target_fasting', '-')} mg/dL")
        col4.metric("Target Post",  f"{p.get('target_postmeal', '-')} mg/dL")

        st.info(f"✅ Profile complete! All tabs are now personalized for **{p.get('name', 'you')}**.")
    else:
        st.warning("⚠️ Please fill and save your profile to personalize all features.")

# ════════════════════════════════════════════════════════
# TAB 1 — RISK ASSESSMENT
# ════════════════════════════════════════════════════════
with tab1:
    p = st.session_state.get("profile", {})

    # Show profile banner if available
    if p:
        st.success(f"👤 Logged in as: **{p.get('name')}** | {p.get('diabetes_type')} | BMI: {p.get('bmi')}")
        
    st.subheader("🔍 Diabetes Risk Assessment")
    st.write("Enter your health metrics to assess your diabetes risk.")

    col1, col2 = st.columns(2)

    with col1:
        glucose    = st.number_input("Glucose Level (mg/dL)",
                              min_value=50, max_value=300,
                              value=117, key="ra_glucose",
                              help="Normal: 70-100 mg/dL fasting")
        bmi        = st.number_input("BMI",
                              min_value=10.0, max_value=70.0,
                              value=float(p.get('bmi', 32.0)),
                              key="ra_bmi",
                              help="Normal: 18.5-24.9")
        age        = st.number_input("Age",
                              min_value=18, max_value=100,
                              value=int(p.get('age', 33)),
                              key="ra_age")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)",
                                  min_value=40, max_value=150,
                                  value=72, key="ra_bp",
                                  help="Normal: 60-80 mm Hg diastolic")

    with col2:
        pregnancies = st.number_input("Number of Pregnancies",
                               min_value=0, max_value=20,
                               value=1, key="ra_preg")
        insulin     = st.number_input("Insulin Level (μU/mL)",
                               min_value=0, max_value=900,
                               value=30, key="ra_insulin",
                               help="Normal fasting: 2-25 μU/mL")
        skin_thickness = st.number_input("Skin Thickness (mm)",
                                  min_value=0, max_value=100,
                                  value=23, key="ra_skin",
                                  help="Triceps skin fold thickness")
        dpf = st.number_input("Diabetes Pedigree Function",
                       min_value=0.0, max_value=3.0,
                       value=0.47, key="ra_dpf",
                       help="Family history score")

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
    # Replace the user_context block with this:
    user_context = ""
    if p:
        user_context = f"""
Patient Profile:
- Name: {p.get('name')}
- Age: {p.get('age')}, Gender: {p.get('gender')}
- Diabetes Status: {p.get('diabetes_type')}
- BMI: {p.get('bmi')}
- Conditions: {', '.join(p.get('conditions', []))}
- Diet: {p.get('diet_pref')}
- Fitness: {p.get('fitness_level')}
Always address them by name and personalize your answers."""

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
    if question := st.chat_input("Ask about diabetes..."): #walrus operator — same thing in one line:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.health_messages.append({
            "role": "user", "content": question
        })

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = call_groq(system, question, history=st.session_state.health_messages)
            st.write(answer)

        st.session_state.health_messages.append({
            "role": "assistant", "content": answer
        })

# ════════════════════════════════════════════════════════
# TAB 3 — DIET PLAN
# ════════════════════════════════════════════════════════
with tab3:
    p = st.session_state.get("profile", {})

    if p:
        st.success(f"👤 Generating plan for: **{p.get('name')}**")
        # Pre-fill from profile
        diet_type = p.get('diet_pref', 'Vegetarian')
        cuisine   = p.get('cuisine', 'Indian')
        allergies = p.get('allergies', [])
        st.info(f"Using your profile: {diet_type} | {cuisine} cuisine | Allergies: {', '.join(allergies)}")

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
                                       ["Vegetarian", "Non-Vegetarian", "Vegan"],
                                       key="tab3_diet_type")
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
    p = st.session_state.get("profile", {})

    if p:
        st.success(f"👤 {p.get('name')}'s Health Tracker")

    tracker_tab1, tracker_tab2, tracker_tab3, tracker_tab4 = st.tabs([
        "🩸 Blood Sugar",
        "💊 Medications",
        "🏃 Exercise",
        "🥗 Food"
    ])

    # ════════════════════════════════════════════════════
    # BLOOD SUGAR TRACKER — unchanged
    # ════════════════════════════════════════════════════
    with tracker_tab1:
        p = st.session_state.get("profile", {})
        if p:
            st.info(f"👤 {p.get('name')}'s targets — Fasting: {p.get('target_fasting')} mg/dL | Post-meal: {p.get('target_postmeal')} mg/dL")

        st.subheader("🩸 Blood Sugar Tracker")

        if "blood_sugar_log" not in st.session_state:
            st.session_state.blood_sugar_log = []

        col1, col2, col3 = st.columns(3)
        with col1:
            bs_date  = st.date_input("Date", value=date.today())
        with col2:
            bs_time  = st.selectbox("When", [
                "Fasting (morning)", "Before breakfast", "After breakfast",
                "Before lunch", "After lunch", "Before dinner",
                "After dinner", "Bedtime"
            ])
        with col3:
            bs_value = st.number_input("Blood Sugar (mg/dL)",
                                        min_value=50, max_value=600, value=120)

        col1, col2 = st.columns(2)
        with col1:
            bs_notes = st.text_input("Notes (optional)",
                                      placeholder="e.g. after exercise...")
        with col2:
            st.write("")
            st.write("")
            if st.button("➕ Log Reading", type="primary"):
                st.session_state.blood_sugar_log.append({
                    "date": str(bs_date), "time": bs_time,
                    "value": bs_value, "notes": bs_notes
                })
                st.success(f"✅ Logged: {bs_value} mg/dL")

        def get_bs_status(value):
            if value < 70:   return "⚠️ Low (Hypoglycemia)", "red"
            elif value <= 100: return "✅ Normal (Fasting)", "green"
            elif value <= 125: return "⚠️ Pre-diabetic range", "orange"
            elif value <= 180: return "⚠️ High (Post-meal)", "orange"
            else:              return "🚨 Very High", "red"

        status, color = get_bs_status(bs_value)
        st.markdown(f"**Status:** :{color}[{status}]")

        with st.expander("📋 Reference Ranges"):
            st.table(pd.DataFrame({
                "Condition": ["Normal fasting", "Pre-diabetes",
                               "Diabetes", "Normal post-meal", "High post-meal"],
                "Range (mg/dL)": ["70-100", "100-125", "126+", "<140", "140+"]
            }))

        st.divider()

        if st.session_state.blood_sugar_log:
            df_log = pd.DataFrame(st.session_state.blood_sugar_log)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df_log))), y=df_log['value'],
                mode='lines+markers', name='Blood Sugar',
                line=dict(color='#6366f1', width=2), marker=dict(size=8)
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="green",
                          annotation_text="Normal max (100)")
            fig.add_hline(y=126, line_dash="dash", line_color="red",
                          annotation_text="Diabetes threshold (126)")
            fig.add_hline(y=180, line_dash="dash", line_color="orange",
                          annotation_text="Post-meal high (180)")
            fig.update_layout(
                title="Blood Sugar Trend",
                xaxis_title="Reading #", yaxis_title="Blood Sugar (mg/dL)",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'), height=400
            )
            st.plotly_chart(fig, use_container_width=True)

            values = df_log['value'].tolist()
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Average",  f"{sum(values)/len(values):.0f} mg/dL")
            s2.metric("Highest",  f"{max(values)} mg/dL")
            s3.metric("Lowest",   f"{min(values)} mg/dL")
            s4.metric("Readings", len(values))

            st.dataframe(df_log, use_container_width=True)
            if st.button("🗑️ Clear Readings"):
                st.session_state.blood_sugar_log = []
                st.rerun()
        else:
            st.info("No readings logged yet.")

    # ════════════════════════════════════════════════════
    # MEDICATION TRACKER — upgraded
    # ════════════════════════════════════════════════════
    with tracker_tab2:
        st.subheader("💊 Medication Tracker")
        st.caption("Track your medications and daily adherence")

        if "medications" not in st.session_state:
            st.session_state.medications = []
        if "med_log" not in st.session_state:
            st.session_state.med_log = {}

        # ── Add Medication ───────────────────────────────
        with st.expander("➕ Add New Medication", expanded=len(st.session_state.medications) == 0):
            col1, col2, col3 = st.columns(3)
            with col1:
                med_name = st.text_input("Medication Name",
                                          placeholder="e.g. Metformin",
                                          key="med_name_input")
            with col2:
                med_dose = st.text_input("Dosage",
                                          placeholder="e.g. 500mg",
                                          key="med_dose_input")
            with col3:
                med_freq = st.selectbox("Frequency", [
                    "Once daily", "Twice daily", "Three times daily",
                    "With meals", "Before meals", "After meals", "At bedtime"
                ], key="med_freq_input")

            med_time = st.multiselect("Reminder Times",
                                       ["Morning", "Afternoon",
                                        "Evening", "Night"],
                                       key="med_time_input")
            med_notes = st.text_input("Notes",
                                       placeholder="e.g. take with food",
                                       key="med_notes_input")

            if st.button("➕ Add Medication", type="primary"):
                if med_name:
                    st.session_state.medications.append({
                        "name":      med_name,
                        "dose":      med_dose,
                        "frequency": med_freq,
                        "times":     med_time,
                        "notes":     med_notes
                    })
                    st.success(f"✅ Added: {med_name}")
                    st.rerun()
                else:
                    st.error("Please enter medication name")

        # ── Today's Medications ──────────────────────────
        if st.session_state.medications:
            st.subheader(f"📅 Today's Medications — {date.today().strftime('%d %b %Y')}")

            today_key = str(date.today())
            if today_key not in st.session_state.med_log:
                st.session_state.med_log[today_key] = {}

            taken_count = 0
            for i, med in enumerate(st.session_state.medications):
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                with col1:
                    st.write(f"💊 **{med['name']}** — {med['dose']}")
                with col2:
                    st.write(f"🕐 {med['frequency']}")
                with col3:
                    taken = st.checkbox(
                        "✅ Taken",
                        value=st.session_state.med_log[today_key].get(med['name'], False),
                        key=f"taken_{today_key}_{i}"
                    )
                    st.session_state.med_log[today_key][med['name']] = taken
                    if taken:
                        taken_count += 1
                with col4:
                    if st.button("🗑️", key=f"del_med_{i}"):
                        st.session_state.medications.pop(i)
                        st.rerun()

            # Adherence today
            total = len(st.session_state.medications)
            adherence = (taken_count / total * 100) if total > 0 else 0
            st.progress(adherence/100,
                        text=f"Today's adherence: {taken_count}/{total} ({adherence:.0f}%)")

            # ── Adherence Chart (last 7 days) ────────────
            if len(st.session_state.med_log) > 1:
                st.divider()
                st.subheader("📈 7-Day Adherence")

                dates      = []
                adherences = []
                for day_key, day_log in sorted(st.session_state.med_log.items())[-7:]:
                    taken_day = sum(1 for v in day_log.values() if v)
                    adh_pct   = (taken_day / total * 100) if total > 0 else 0
                    dates.append(day_key)
                    adherences.append(adh_pct)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=dates, y=adherences,
                    marker_color=['#22c55e' if a >= 80 else '#f59e0b'
                                  if a >= 50 else '#ef4444' for a in adherences],
                    name='Adherence %'
                ))
                fig.update_layout(
                    title="Medication Adherence (Last 7 Days)",
                    yaxis=dict(range=[0, 100], title="Adherence %"),
                    xaxis_title="Date",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            # Download
            med_text = f"Medication List — {date.today()}\n" + "="*30 + "\n\n"
            for med in st.session_state.medications:
                med_text += f"• {med['name']} — {med['dose']} — {med['frequency']}\n"
                if med['notes']:
                    med_text += f"  Notes: {med['notes']}\n"
            st.download_button("⬇️ Download List", data=med_text,
                                file_name="medications.txt", mime="text/plain")
        else:
            st.info("No medications added yet. Add your first medication above.")

        st.warning("⚠️ Always follow your doctor's prescription.")

    # ════════════════════════════════════════════════════
    # EXERCISE TRACKER — upgraded
    # ════════════════════════════════════════════════════
    with tracker_tab3:
        st.subheader("🏃 Exercise Tracker")
        st.caption("Log your daily exercise and track your progress")

        if "exercise_log" not in st.session_state:
            st.session_state.exercise_log = []

        # ── Log Exercise ─────────────────────────────────
        st.subheader("➕ Log Today's Exercise")

        col1, col2, col3 = st.columns(3)
        with col1:
            ex_date = st.date_input("Date",
                                     value=date.today(),
                                     key="ex_date")
            ex_type = st.selectbox("Exercise Type", [
                "Walking", "Jogging", "Running", "Cycling",
                "Swimming", "Yoga", "Strength Training",
                "Stretching", "Dancing", "Other"
            ])
        with col2:
            ex_duration = st.number_input("Duration (minutes)",
                                           min_value=5, max_value=300,
                                           value=30, key="ex_duration")
            ex_intensity = st.selectbox("Intensity",
                                         ["Light", "Moderate", "Intense"])
        with col3:
            ex_calories = st.number_input("Calories Burned (approx)",
                                           min_value=0, max_value=2000,
                                           value=150, key="ex_calories")
            ex_bs_before = st.number_input("Blood Sugar Before (optional)",
                                            min_value=0, max_value=600,
                                            value=0, key="ex_bs_before",
                                            help="0 = not measured")
            ex_bs_after  = st.number_input("Blood Sugar After (optional)",
                                            min_value=0, max_value=600,
                                            value=0, key="ex_bs_after",
                                            help="0 = not measured")

        ex_notes = st.text_input("Notes", placeholder="How did it feel?",
                                  key="ex_notes")

        if st.button("➕ Log Exercise", type="primary"):
            st.session_state.exercise_log.append({
                "date":       str(ex_date),
                "type":       ex_type,
                "duration":   ex_duration,
                "intensity":  ex_intensity,
                "calories":   ex_calories,
                "bs_before":  ex_bs_before,
                "bs_after":   ex_bs_after,
                "notes":      ex_notes
            })
            st.success(f"✅ Logged: {ex_duration} min {ex_type}")
            st.rerun()

        # ── Stats + Chart ────────────────────────────────
        if st.session_state.exercise_log:
            df_ex = pd.DataFrame(st.session_state.exercise_log)

            # Summary stats
            total_mins = df_ex['duration'].sum()
            total_cals = df_ex['calories'].sum()
            avg_mins   = df_ex['duration'].mean()
            streak     = 0
            dates_set  = set(df_ex['date'].tolist())
            check_date = date.today()
            while str(check_date) in dates_set:
                streak    += 1
                check_date = check_date.replace(day=check_date.day - 1)

            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Total Minutes",   f"{total_mins}")
            e2.metric("Calories Burned", f"{total_cals}")
            e3.metric("Avg per Session", f"{avg_mins:.0f} min")
            e4.metric("🔥 Streak",       f"{streak} days")

            st.divider()

            # Weekly chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_ex['date'].tolist()[-7:],
                y=df_ex['duration'].tolist()[-7:],
                marker_color='#6366f1',
                name='Duration (min)'
            ))
            fig.add_hline(y=30, line_dash="dash",
                          line_color="green",
                          annotation_text="Recommended 30 min")
            fig.update_layout(
                title="Exercise Log (Last 7 Sessions)",
                xaxis_title="Date",
                yaxis_title="Duration (minutes)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            # Blood sugar impact
            bs_data = df_ex[(df_ex['bs_before'] > 0) & (df_ex['bs_after'] > 0)]
            if len(bs_data) > 0:
                st.subheader("📉 Exercise Impact on Blood Sugar")
                for _, row in bs_data.iterrows():
                    diff  = row['bs_after'] - row['bs_before']
                    color = "green" if diff < 0 else "red"
                    st.markdown(
                        f"**{row['date']}** — {row['type']}: "
                        f"{row['bs_before']} → {row['bs_after']} "
                        f"(:{color}[{diff:+.0f} mg/dL])"
                    )

            # Log table
            with st.expander("📋 Full Exercise History"):
                st.dataframe(df_ex, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                ex_text = f"Exercise Log\n{'='*30}\n\n"
                for _, row in df_ex.iterrows():
                    ex_text += f"{row['date']} — {row['type']} — {row['duration']}min — {row['calories']} cal\n"
                st.download_button("⬇️ Download Log",
                                    data=ex_text,
                                    file_name="exercise_log.txt",
                                    mime="text/plain")
            with col2:
                if st.button("🗑️ Clear Exercise Log"):
                    st.session_state.exercise_log = []
                    st.rerun()
        else:
            st.info("No exercise logged yet. Log your first session above!")

    # ════════════════════════════════════════════════════
    # FOOD TRACKER — new
    # ════════════════════════════════════════════════════
    with tracker_tab4:
        st.subheader("🥗 Food Tracker")
        st.caption("Track meals, carbs and glycemic index")

        if "food_log" not in st.session_state:
            st.session_state.food_log = []

        # Glycemic index reference
        GI_DATA = {
            "White rice":        {"gi": 72, "carbs": 45, "category": "High GI"},
            "Brown rice":        {"gi": 50, "carbs": 42, "category": "Medium GI"},
            "White bread":       {"gi": 75, "carbs": 15, "category": "High GI"},
            "Whole wheat bread": {"gi": 50, "carbs": 12, "category": "Medium GI"},
            "Oats":              {"gi": 55, "carbs": 27, "category": "Medium GI"},
            "Apple":             {"gi": 36, "carbs": 25, "category": "Low GI"},
            "Banana":            {"gi": 51, "carbs": 27, "category": "Medium GI"},
            "Idli":              {"gi": 35, "carbs": 20, "category": "Low GI"},
            "Dosa":              {"gi": 52, "carbs": 30, "category": "Medium GI"},
            "Dal":               {"gi": 25, "carbs": 20, "category": "Low GI"},
            "Chapati":           {"gi": 52, "carbs": 18, "category": "Medium GI"},
            "Poha":              {"gi": 55, "carbs": 35, "category": "Medium GI"},
            "Upma":              {"gi": 50, "carbs": 30, "category": "Medium GI"},
            "Milk":              {"gi": 31, "carbs": 12, "category": "Low GI"},
            "Curd/Yogurt":       {"gi": 36, "carbs": 6,  "category": "Low GI"},
            "Potato":            {"gi": 78, "carbs": 30, "category": "High GI"},
            "Sweet potato":      {"gi": 55, "carbs": 24, "category": "Medium GI"},
            "Watermelon":        {"gi": 72, "carbs": 8,  "category": "High GI"},
            "Mango":             {"gi": 56, "carbs": 25, "category": "Medium GI"},
        }

        # ── Log Meal ─────────────────────────────────────
        st.subheader("➕ Log a Meal")

        col1, col2, col3 = st.columns(3)
        with col1:
            food_date  = st.date_input("Date", value=date.today(), key="food_date")
            meal_type  = st.selectbox("Meal", [
                "Breakfast", "Morning Snack", "Lunch",
                "Evening Snack", "Dinner", "Late Night"
            ])
        with col2:
            food_item  = st.text_input("Food Item",
                                        placeholder="e.g. Brown rice, Dal...")
            portion    = st.selectbox("Portion Size",
                                       ["Small (0.5x)", "Medium (1x)",
                                        "Large (1.5x)", "Extra Large (2x)"])
        with col3:
            carbs      = st.number_input("Carbs (g)",
                                          min_value=0, max_value=500, value=30,
                                          key="food_carbs")
            calories   = st.number_input("Calories (approx)",
                                          min_value=0, max_value=2000, value=200,
                                          key="food_calories")

        # GI lookup
        selected_gi_food = st.selectbox(
            "Or lookup GI from common foods:",
            ["-- Select --"] + list(GI_DATA.keys())
        )

        if selected_gi_food != "-- Select --":
            gi_info = GI_DATA[selected_gi_food]
            g1, g2, g3 = st.columns(3)
            g1.metric("Glycemic Index", gi_info['gi'])
            g2.metric("Carbs per serving", f"{gi_info['carbs']}g")
            g3.metric("Category", gi_info['category'])

        if st.button("➕ Log Meal", type="primary"):
            gi_val = GI_DATA.get(selected_gi_food, {}).get('gi', 0) \
                     if selected_gi_food != "-- Select --" else 0
            gi_cat = GI_DATA.get(selected_gi_food, {}).get('category', 'Unknown') \
                     if selected_gi_food != "-- Select --" else 'Unknown'

            st.session_state.food_log.append({
                "date":     str(food_date),
                "meal":     meal_type,
                "food":     food_item if food_item else selected_gi_food,
                "portion":  portion,
                "carbs":    carbs,
                "calories": calories,
                "gi":       gi_val,
                "gi_cat":   gi_cat
            })
            st.success(f"✅ Logged: {meal_type} — {food_item or selected_gi_food}")
            st.rerun()

        # ── GI Reference Table ───────────────────────────
        with st.expander("📋 Glycemic Index Guide"):
            gi_df = pd.DataFrame([
                {"Food": k, "GI": v['gi'],
                 "Carbs/serving": f"{v['carbs']}g",
                 "Category": v['category']}
                for k, v in GI_DATA.items()
            ]).sort_values('GI')
            st.dataframe(gi_df, use_container_width=True)

        # ── Today's Summary ──────────────────────────────
        if st.session_state.food_log:
            df_food   = pd.DataFrame(st.session_state.food_log)
            today_str = str(date.today())
            today_food = df_food[df_food['date'] == today_str]

            st.divider()
            st.subheader(f"📊 Today's Summary — {date.today().strftime('%d %b %Y')}")

            if not today_food.empty:
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Total Carbs",    f"{today_food['carbs'].sum()}g")
                f2.metric("Total Calories", today_food['calories'].sum())
                f3.metric("Meals Logged",   len(today_food))
                avg_gi = today_food[today_food['gi'] > 0]['gi'].mean()
                f4.metric("Avg GI",         f"{avg_gi:.0f}" if avg_gi > 0 else "N/A")

                # Flag high GI foods
                high_gi = today_food[today_food['gi'] >= 70]
                if not high_gi.empty:
                    st.warning(f"⚠️ High GI foods today: {', '.join(high_gi['food'].tolist())}")

                # Carbs chart
                fig = go.Figure(go.Bar(
                    x=today_food['meal'],
                    y=today_food['carbs'],
                    marker_color=['#22c55e' if c < 30 else
                                  '#f59e0b' if c < 60 else
                                  '#ef4444' for c in today_food['carbs']],
                    name='Carbs (g)'
                ))
                fig.update_layout(
                    title="Carbs per Meal Today",
                    xaxis_title="Meal",
                    yaxis_title="Carbs (g)",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

            # Full log
            with st.expander("📋 Full Food History"):
                st.dataframe(df_food, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                food_text = f"Food Log\n{'='*30}\n\n"
                for _, row in df_food.iterrows():
                    food_text += f"{row['date']} — {row['meal']}: {row['food']} — {row['carbs']}g carbs — {row['calories']} cal\n"
                st.download_button("⬇️ Download Food Log",
                                    data=food_text,
                                    file_name="food_log.txt",
                                    mime="text/plain")
            with col2:
                if st.button("🗑️ Clear Food Log"):
                    st.session_state.food_log = []
                    st.rerun()
        else:
            st.info("No meals logged yet. Log your first meal above!")



# ════════════════════════════════════════════════════════
# TAB 5 — DASHBOARD
# ════════════════════════════════════════════════════════
with tab5:
    p = st.session_state.get("profile", {})

    if p:
        st.markdown(f"## 👋 Welcome back, {p.get('name')}!")
    else:
        st.markdown("## 📈 Health Dashboard")
        st.warning("Complete your profile for personalized insights")

    st.caption(f"Today: {date.today().strftime('%A, %d %B %Y')}")
    st.divider()

    # ── Overview Metrics ─────────────────────────────────
    st.subheader("📊 Today's Overview")

    bs_log       = st.session_state.get("blood_sugar_log", [])
    med_list     = st.session_state.get("medications", [])
    med_log      = st.session_state.get("med_log", {})
    exercise_log = st.session_state.get("exercise_log", [])
    food_log     = st.session_state.get("food_log", [])

    today_str    = str(date.today())
    today_bs     = [r for r in bs_log if r['date'] == today_str]
    today_ex     = [r for r in exercise_log if r['date'] == today_str]
    today_food   = [r for r in food_log if r['date'] == today_str]
    today_meds   = med_log.get(today_str, {})
    meds_taken   = sum(1 for v in today_meds.values() if v)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🩸 BS Readings",    len(today_bs),
              f"Avg: {sum(r['value'] for r in today_bs)/len(today_bs):.0f}" if today_bs else "No readings")
    c2.metric("💊 Meds Taken",     f"{meds_taken}/{len(med_list)}",
              "✅ All done!" if meds_taken == len(med_list) and len(med_list) > 0 else "")
    c3.metric("🏃 Exercise",       f"{sum(r['duration'] for r in today_ex)} min",
              f"{len(today_ex)} session(s)")
    c4.metric("🥗 Meals Logged",   len(today_food),
              f"{sum(r['carbs'] for r in today_food)}g carbs")
    c5.metric("🎯 Risk Level",
              st.session_state.get('risk_level', 'Not assessed'), "")

    st.divider()

    # ── A1C Estimator ────────────────────────────────────
    st.subheader("🔬 A1C Estimator")
    st.caption("Estimated from your average blood sugar readings")

    if bs_log:
        values  = [r['value'] for r in bs_log]
        avg_bs  = sum(values) / len(values)
        a1c_est = (avg_bs + 46.7) / 28.7

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Blood Sugar", f"{avg_bs:.0f} mg/dL",
                    f"from {len(values)} readings")
        col2.metric("Estimated A1C",       f"{a1c_est:.1f}%")

        if a1c_est < 5.7:
            col3.metric("A1C Category", "Normal ✅")
        elif a1c_est < 6.5:
            col3.metric("A1C Category", "Pre-diabetes ⚠️")
        else:
            col3.metric("A1C Category", "Diabetes range 🚨")

        # A1C gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=a1c_est,
            number={'suffix': "%"},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Estimated A1C", 'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [4, 12],
                         'tickcolor': 'white',
                         'tickfont': {'color': 'white'}},
                'bar':  {'color': '#6366f1'},
                'steps': [
                    {'range': [4, 5.7],  'color': '#22c55e'},
                    {'range': [5.7, 6.5],'color': '#f59e0b'},
                    {'range': [6.5, 12], 'color': '#ef4444'}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': a1c_est
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white'},
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ About A1C"):
            st.write("""
**A1C (HbA1c)** measures your average blood sugar over 2-3 months.

| A1C Level | Category |
|-----------|----------|
| Below 5.7% | Normal |
| 5.7% - 6.4% | Pre-diabetes |
| 6.5% and above | Diabetes |

⚠️ This is an **estimate** based on your logged readings.
Get a proper A1C test from your doctor for accurate results.
            """)
    else:
        st.info("Log blood sugar readings to see your estimated A1C")

    st.divider()

    # ── Weekly Charts ─────────────────────────────────────
    if bs_log or exercise_log:
        st.subheader("📈 Weekly Trends")

        col1, col2 = st.columns(2)

        with col1:
            if bs_log:
                last7_bs = [r for r in bs_log][-7:]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(last7_bs))),
                    y=[r['value'] for r in last7_bs],
                    mode='lines+markers',
                    line=dict(color='#6366f1', width=2),
                    name='Blood Sugar'
                ))
                fig.add_hline(y=126, line_dash="dash",
                              line_color="red", annotation_text="126")
                fig.update_layout(
                    title="Blood Sugar Trend",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if exercise_log:
                last7_ex = exercise_log[-7:]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[r['date'] for r in last7_ex],
                    y=[r['duration'] for r in last7_ex],
                    marker_color='#22c55e',
                    name='Exercise (min)'
                ))
                fig.add_hline(y=30, line_dash="dash",
                              line_color="yellow",
                              annotation_text="30 min goal")
                fig.update_layout(
                    title="Exercise Duration",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Health Report Download ────────────────────────────
    st.subheader("📄 Download Health Report")
    st.write("Get a complete PDF report of your health data.")

    if st.button("📄 Generate Health Report", type="primary"):
        risk_data = None
        if 'risk_level' in st.session_state:
            risk_data = {
                'level': st.session_state.risk_level,
                'prob':  st.session_state.risk_prob
            }

        with st.spinner("Generating your health report..."):
            pdf_buffer = generate_health_report(
                profile=st.session_state.get("profile"),
                risk_data=risk_data,
                bs_log=bs_log,
                med_log=med_list,
                exercise_log=exercise_log,
                food_log=food_log
            )

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_buffer,
            file_name=f"health_report_{date.today()}.pdf",
            mime="application/pdf"
        )
        st.success("✅ Report ready! Click above to download.")