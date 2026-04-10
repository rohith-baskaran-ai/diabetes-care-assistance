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

def call_groq(system, user, history=None, max_tokens=1000):
    messages = [{"role": "system", "content": system}]
    
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
        max_tokens=max_tokens  # ← use parameter
    )
    return response.choices[0].message.content

def analyze_prescription(image_bytes, image_type="jpeg"):
    import base64
    import json as json_lib

    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You are a medical prescription reader.
Extract all medicines from this prescription image.
Return ONLY valid JSON — no extra text:
{
  "medicines": [
    {
      "name": "medicine name",
      "dose": "total dose",
      "morning": "dose in morning or empty string",
      "noon": "dose at noon or empty string",
      "evening": "dose in evening or empty string",
      "night": "dose at night or empty string",
      "duration": "how long e.g. 30 days",
      "route": "oral/SC/IV/VG etc",
      "instructions": "special instructions"
    }
  ],
  "doctor_name": "doctor name if visible",
  "patient_name": "patient name if visible",
  "date": "date if visible"
}

For each medicine extract exact dose per time slot.
If a slot is empty or has - put empty string.
Example for Actrapid 20U morning 20U noon 20U night:
{
  "name": "Actrapid",
  "dose": "20 Units",
  "morning": "20 Units",
  "noon": "20 Units",
  "evening": "",
  "night": "20 Units",
  "duration": "30 days",
  "route": "SC",
  "instructions": ""
}"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000
    )

    raw = response.choices[0].message.content

    try:
        clean = raw.strip()
        if "```json" in clean:
            clean = clean.split("```json")[1].split("```")[0]
        elif "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]
        return json_lib.loads(clean)
    except:
        return None

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

    # show profile banner
    if p:
        st.success(
            f"👤 {p.get('name')} | {p.get('diabetes_type')} | "
            f"BMI: {p.get('bmi')} | {p.get('diet_pref')} | "
            f"{p.get('cuisine')} cuisine"
        )

    st.subheader("🥗 Personalized Diet Plan")

    # initialize session state
    if "diet_chat"        not in st.session_state:
        st.session_state.diet_chat        = []
    if "diet_preferences" not in st.session_state:
        st.session_state.diet_preferences = {}
    if "weekly_plan"      not in st.session_state:
        st.session_state.weekly_plan      = []
    if "plan_generated"   not in st.session_state:
        st.session_state.plan_generated   = False
    if "plan_ready"       not in st.session_state:
        st.session_state.plan_ready       = False

    
    # debug — add here after initialization
    st.write(f"DEBUG — generated: {st.session_state.plan_generated} | "
         f"ready: {st.session_state.plan_ready} | "
         f"plan: {len(st.session_state.weekly_plan)} days")

    # ── STEP 1: Setup Chat ───────────────────────────────
    if not st.session_state.plan_generated:

        st.markdown("### 💬 Step 1 — Tell me your preferences")
        st.caption("Chat with the AI to set up your personalized meal plan")

        # build system prompt for diet setup
        risk_level = st.session_state.get('risk_level', 'Medium Risk')
        risk_prob  = st.session_state.get('risk_prob', 0.5)

        setup_system = f"""You are a friendly diabetes nutritionist helping set up
a personalized meal plan for a patient.

Patient Profile:
- Name: {p.get('name', 'Patient')}
- Age: {p.get('age', 'Unknown')}
- Diabetes Status: {p.get('diabetes_type', 'Unknown')}
- BMI: {p.get('bmi', 'Unknown')}
- Risk Level: {risk_level} ({risk_prob*100:.0f}%)
- Diet: {p.get('diet_pref', 'Vegetarian')}
- Cuisine: {p.get('cuisine', 'Indian')}
- Allergies: {', '.join(p.get('allergies', ['None']))}
- Conditions: {', '.join(p.get('conditions', ['None']))}

Your job:
1. Greet them and ask these 6 questions ONE AT A TIME:
   Q1: How many meals per day? (suggest 4-5 for diabetics)
   Q2: Wake up and sleep time?
   Q3: Favourite foods to include?
   Q4: Foods to absolutely avoid?
   Q5: How much cooking time available per meal?
   Q6: Any specific health goal this week?

2. After all answers — summarize preferences clearly
3. End with: "Ready to generate your 7-day plan? 
   Type 'yes' or click Generate below!"

Be friendly, give suggestions based on their profile.
Keep responses concise — max 3 sentences per message."""

        # show welcome message on first load
        if not st.session_state.diet_chat:
            with st.chat_message("assistant"):
                welcome = f"""Hi {p.get('name', 'there')}! 👋 Let's create your 
personalized 7-day diabetes meal plan.

I'll ask you a few quick questions to make it perfect for you.
Based on your {p.get('diet_pref', 'diet')} preference and 
{risk_level}, I'll suggest the best options.

**How many meals per day do you prefer?**
For diabetics, I recommend **4-5 smaller meals** to keep 
blood sugar stable. What works for you?"""
                st.write(welcome)
            st.session_state.diet_chat.append({
                "role": "assistant", "content": welcome
            })

        # show chat history
        for msg in st.session_state.diet_chat[1:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # chat input
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Start Over", key="diet_reset"):
                st.session_state.diet_chat        = []
                st.session_state.diet_preferences = {}
                st.session_state.plan_generated   = False
                st.session_state.plan_ready       = False
                st.rerun()

        if user_msg := st.chat_input("Type your answer...",
                                      key="diet_chat_input"):
            # show user message
            with st.chat_message("user"):
                st.write(user_msg)
            st.session_state.diet_chat.append({
                "role": "user", "content": user_msg
            })

            # get LLM response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = call_groq(
                        system=setup_system,
                        user=user_msg,
                        history=st.session_state.diet_chat[:-1]
                    )
                st.write(response)
            st.session_state.diet_chat.append({
                "role": "assistant", "content": response
            })

            # check if patient said yes/ready
            if any(word in user_msg.lower() for word in
                   ["yes", "ready", "generate", "ok", "sure", "proceed"]):
                st.session_state.plan_ready = True
                st.rerun()

        # generate button
        # show generate button after enough conversation
        if len(st.session_state.diet_chat) >= 6 or st.session_state.plan_ready:
            st.divider()
            st.info("✅ Preferences collected! Ready to generate your plan?")
            if st.button("🥗 Generate My 7-Day Plan",
                         type="primary",
                         key="generate_plan_btn"):
                st.session_state.plan_generated = True
                st.session_state.plan_ready     = True
                st.rerun()

    # ── STEP 2: Generate + Parse Plan ───────────────────
    elif st.session_state.plan_generated and not st.session_state.weekly_plan:
        p          = st.session_state.get("profile", {})
        risk_level = st.session_state.get('risk_level', 'Medium Risk')
        risk_prob  = st.session_state.get('risk_prob', 0.5)

        # extract preferences from chat
        chat_summary = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in st.session_state.diet_chat
        ])

        with st.spinner("🧠 Creating your personalized plan..."):

            # Step 2a: extract preferences
            prefs_raw = call_groq(
                system="""Extract meal preferences from this conversation.
Return JSON only:
{
  "meals_per_day": 4,
  "wake_time": "7am",
  "sleep_time": "10pm",
  "favourite_foods": ["idli", "dal"],
  "avoid_foods": ["maida", "fried"],
  "cooking_time": "30 minutes",
  "weekly_goal": "lower blood sugar",
  "meal_names": ["Breakfast", "Lunch", "Evening Snack", "Dinner"]
}""",
                user=f"Extract from:\n{chat_summary}",
                max_tokens=4000 
            )

            try:
                import json as json_lib
                prefs_clean = prefs_raw.strip()
                if "```json" in prefs_clean:
                    prefs_clean = prefs_clean.split("```json")[1].split("```")[0]
                elif "```" in prefs_clean:
                    prefs_clean = prefs_clean.split("```")[1]
                    if prefs_clean.startswith("json"):
                        prefs_clean = prefs_clean[4:]
                start = prefs_clean.find("{")
                end   = prefs_clean.rfind("}") + 1
                if start != -1 and end > start:
                    prefs_clean = prefs_clean[start:end]
                prefs = json_lib.loads(prefs_clean)
            except:
                prefs = {
                    "meals_per_day": 4,
                    "meal_names": ["Breakfast", "Lunch",
                                   "Evening Snack", "Dinner"],
                    "favourite_foods": [],
                    "avoid_foods": []
                }

            st.session_state.diet_preferences = prefs
            meal_names = prefs.get("meal_names",
                                    ["Breakfast", "Lunch",
                                     "Evening Snack", "Dinner"])

            # Step 2b: generate structured plan
            days = ["Monday","Tuesday","Wednesday",
                    "Thursday","Friday","Saturday","Sunday"]

            plan_prompt = f"""Create a 7-day diabetes meal plan.

Patient:
- Diet: {p.get('diet_pref', 'Vegetarian')}
- Cuisine: {p.get('cuisine', 'Indian')}
- Risk: {risk_level} ({risk_prob*100:.0f}%)
- BMI: {p.get('bmi', 'Unknown')}
- Allergies: {', '.join(p.get('allergies', ['None']))}
- Favourite foods: {', '.join(prefs.get('favourite_foods', []))}
- Avoid: {', '.join(prefs.get('avoid_foods', []))}
- Meals per day: {prefs.get('meals_per_day', 4)}
- Meal times: {', '.join(meal_names)}
- Goal: {prefs.get('weekly_goal', 'manage blood sugar')}

Return ONLY valid JSON — no extra text:
{{
  "days": [
    {{
      "day": 1,
      "day_name": "Monday",
      "meals": [
        {{
          "meal_type": "Breakfast",
          "time": "8:00 AM",
          "items": [
            {{"name": "Idli", "quantity": "2 pieces",
              "carbs": 20, "calories": 100}},
            {{"name": "Sambar", "quantity": "1 bowl",
              "carbs": 10, "calories": 60}}
          ],
          "total_carbs": 30,
          "total_calories": 160,
          "notes": "Low GI breakfast"
        }}
      ]
    }}
  ]
}}

Include all 7 days with {prefs.get('meals_per_day', 4)} 
meals each. Vary foods daily. Focus on low GI foods."""

            plan_raw = call_groq(
                system="You are a diabetes nutritionist. Return only valid JSON.",
                user=plan_prompt,
                max_tokens=4000  # ← add this
            )

            # parse plan JSON
            try:
                plan_clean = plan_raw.strip()

                # remove markdown code blocks
                if "```json" in plan_clean:
                    plan_clean = plan_clean.split("```json")[1].split("```")[0]
                elif "```" in plan_clean:
                    plan_clean = plan_clean.split("```")[1]
                    if plan_clean.startswith("json"):
                        plan_clean = plan_clean[4:]

                # find JSON object in response
                start = plan_clean.find("{")
                end   = plan_clean.rfind("}") + 1
                if start != -1 and end > start:
                    plan_clean = plan_clean[start:end]

                plan_data = json_lib.loads(plan_clean)
                days_plan = plan_data.get("days", [])

            except Exception as e:
                st.error(f"Parse error: {e}")
                st.code(plan_raw[:500])  # show what LLM returned for debugging
                days_plan = []

            # add tracking fields to each meal
            for day in days_plan:
                for meal in day.get("meals", []):
                    meal["status"] = "pending"
                    meal["actual_items"] = []
                    meal["modified_notes"] = ""

            st.session_state.weekly_plan = days_plan

        if days_plan:
            st.success("✅ Your 7-day plan is ready!")
            st.rerun()
        else:
            st.error("Failed to generate plan. Please try again.")
            st.session_state.plan_generated = False

    # ── STEP 3: Weekly Tracker ───────────────────────────
    elif st.session_state.weekly_plan:
        plan = st.session_state.weekly_plan

        # header + reset button
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.markdown("### 📅 Your 7-Day Meal Tracker")
        with col2:
            if st.button("🔄 New Plan", key="new_plan"):
                st.session_state.diet_chat        = []
                st.session_state.diet_preferences = {}
                st.session_state.weekly_plan      = []
                st.session_state.plan_generated   = False
                st.session_state.plan_ready       = False
                st.rerun()
        with col3:
            # calculate overall adherence for download
            all_meals   = [m for d in plan for m in d.get("meals", [])]
            followed    = [m for m in all_meals if m["status"] == "followed"]
            modified    = [m for m in all_meals if m["status"] == "modified"]
            skipped     = [m for m in all_meals if m["status"] == "skipped"]
            pending     = [m for m in all_meals if m["status"] == "pending"]
            total_done  = len(followed) + len(modified)
            total_meals = len(all_meals)
            adherence   = (total_done / total_meals * 100) if total_meals > 0 else 0

        # overall adherence bar
        st.progress(
            adherence/100,
            text=f"Overall adherence: {adherence:.0f}% "
                 f"({total_done}/{total_meals} meals) | "
                 f"✅ {len(followed)} followed | "
                 f"✏️ {len(modified)} modified | "
                 f"❌ {len(skipped)} skipped"
        )

        st.divider()

        # show each day
        for day_idx, day in enumerate(plan):
            day_meals   = day.get("meals", [])
            day_followed = sum(1 for m in day_meals
                               if m["status"] in ["followed","modified"])
            day_total    = len(day_meals)
            day_pct      = (day_followed/day_total*100) if day_total > 0 else 0

            # day color based on adherence
            if day_pct == 100:   day_emoji = "🟢"
            elif day_pct >= 50:  day_emoji = "🟡"
            elif day_pct > 0:    day_emoji = "🟠"
            else:                day_emoji = "⚪"

            with st.expander(
                f"{day_emoji} Day {day['day']} — {day['day_name']} "
                f"({day_followed}/{day_total} meals done | {day_pct:.0f}%)",
                expanded=day_idx == 0
            ):
                for meal_idx, meal in enumerate(day_meals):
                    st.markdown(f"#### 🍽️ {meal['meal_type']} — {meal.get('time','')}")

                    # show planned items
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown("**Planned:**")
                        for item in meal.get("items", []):
                            st.write(
                                f"• {item['name']} ({item['quantity']}) — "
                                f"{item.get('carbs',0)}g carbs, "
                                f"{item.get('calories',0)} cal"
                            )
                        st.caption(
                            f"Total: {meal.get('total_carbs',0)}g carbs | "
                            f"{meal.get('total_calories',0)} cal"
                        )
                        if meal.get("notes"):
                            st.caption(f"💡 {meal['notes']}")

                    with col2:
                        status = meal["status"]
                        if status == "followed":
                            st.success("✅ Followed")
                        elif status == "modified":
                            st.warning("✏️ Modified")
                        elif status == "skipped":
                            st.error("❌ Skipped")
                        else:
                            st.info("⏳ Pending")

                    # action buttons
                    key_base = f"d{day_idx}_m{meal_idx}"
                    b1, b2, b3, b4 = st.columns(4)

                    with b1:
                        if st.button("✅ Followed",
                                     key=f"follow_{key_base}",
                                     type="primary" if status == "followed"
                                     else "secondary"):
                            plan[day_idx]["meals"][meal_idx]["status"] = "followed"
                            plan[day_idx]["meals"][meal_idx]["actual_items"] = []
                            plan[day_idx]["meals"][meal_idx]["modified_notes"] = ""
                            st.session_state.weekly_plan = plan
                            st.rerun()

                    with b2:
                        if st.button("✏️ Modified",
                                     key=f"modify_{key_base}"):
                            plan[day_idx]["meals"][meal_idx]["status"] = "modified"
                            st.session_state.weekly_plan = plan
                            st.rerun()

                    with b3:
                        if st.button("❌ Skipped",
                                     key=f"skip_{key_base}"):
                            plan[day_idx]["meals"][meal_idx]["status"] = "skipped"
                            st.session_state.weekly_plan = plan
                            st.rerun()

                    with b4:
                        if st.button("↩️ Reset",
                                     key=f"reset_{key_base}"):
                            plan[day_idx]["meals"][meal_idx]["status"] = "pending"
                            plan[day_idx]["meals"][meal_idx]["modified_notes"] = ""
                            st.session_state.weekly_plan = plan
                            st.rerun()

                    # modified notes input
                    if meal["status"] == "modified":
                        mod_note = st.text_input(
                            "What did you have instead?",
                            value=meal.get("modified_notes", ""),
                            key=f"modnote_{key_base}",
                            placeholder="e.g. Had upma instead of idli..."
                        )
                        if mod_note != meal.get("modified_notes", ""):
                            plan[day_idx]["meals"][meal_idx]["modified_notes"] = mod_note
                            st.session_state.weekly_plan = plan

                    st.divider()

        # ── Adherence Chart ──────────────────────────────
        st.subheader("📊 Weekly Adherence Chart")

        day_names  = [d['day_name'][:3] for d in plan]
        day_adh    = []
        for day in plan:
            meals   = day.get("meals", [])
            done    = sum(1 for m in meals
                         if m["status"] in ["followed","modified"])
            total   = len(meals)
            day_adh.append((done/total*100) if total > 0 else 0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=day_names,
            y=day_adh,
            marker_color=[
                '#22c55e' if a == 100 else
                '#f59e0b' if a >= 50 else
                '#ef4444' if a > 0 else
                '#64748b'
                for a in day_adh
            ],
            name='Adherence %'
        ))
        fig.add_hline(y=80, line_dash="dash",
                      line_color="white",
                      annotation_text="80% target")
        fig.update_layout(
            title="Daily Meal Adherence",
            yaxis=dict(range=[0,100], title="Adherence %"),
            xaxis_title="Day",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Best / Worst Day ─────────────────────────────
        if any(a > 0 for a in day_adh):
            best_idx  = day_adh.index(max(day_adh))
            worst_idx = day_adh.index(min(day_adh))
            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Best Day",
                        plan[best_idx]['day_name'],
                        f"{day_adh[best_idx]:.0f}%")
            col2.metric("📉 Needs Work",
                        plan[worst_idx]['day_name'],
                        f"{day_adh[worst_idx]:.0f}%")
            col3.metric("📊 Week Average",
                        f"{sum(day_adh)/len(day_adh):.0f}%")

        # ── Download Plan ────────────────────────────────
        st.divider()
        plan_text  = f"7-Day Meal Plan\n{'='*40}\n\n"
        for day in plan:
            plan_text += f"\n{day['day_name']}\n{'-'*20}\n"
            for meal in day.get("meals", []):
                plan_text += f"\n{meal['meal_type']} ({meal.get('time','')})\n"
                for item in meal.get("items", []):
                    plan_text += (f"  • {item['name']} "
                                  f"({item['quantity']})\n")
                plan_text += (f"  Total: {meal.get('total_carbs',0)}g carbs | "
                              f"{meal.get('total_calories',0)} cal\n")
                if meal["status"] != "pending":
                    plan_text += f"  Status: {meal['status']}\n"
                if meal.get("modified_notes"):
                    plan_text += f"  Modified: {meal['modified_notes']}\n"

        st.download_button(
            "⬇️ Download Plan + Tracker",
            data=plan_text,
            file_name="meal_plan_tracker.txt",
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
        if st.button("Clear prescription cache", key="clear_cache"):
            if "prescription_result" in st.session_state:
                del st.session_state.prescription_result
            st.rerun()

        st.subheader("💊 Medication Tracker")
        st.caption("Track your medications and daily adherence")

        if "medications" not in st.session_state:
            st.session_state.medications = []
        if "med_log" not in st.session_state:
            st.session_state.med_log = {}

        # ── Add Medication ───────────────────────────────
        # ── Upload Prescription ──────────────────────────────
        st.subheader("📋 Upload Prescription")
        st.caption("Upload a photo or PDF of your prescription — AI will extract medicines automatically")

        uploaded_prescription = st.file_uploader(
            "Upload prescription image",
            type=["jpg", "jpeg", "png"],
            key="prescription_upload"
        )

        if uploaded_prescription:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(uploaded_prescription, caption="Uploaded Prescription",
                         use_column_width=True)

            with col2:
                if st.button("🔍 Analyze Prescription", type="primary",
                             key="analyze_btn"):
                    with st.spinner("🤖 Reading prescription..."):
                        image_bytes = uploaded_prescription.read()
                        image_type  = uploaded_prescription.type.split("/")[1]
                        result      = analyze_prescription(image_bytes, image_type)

                    if result:
                        st.session_state.prescription_result = result
                        st.success("✅ Prescription analyzed!")
                        # debug — show raw result
                        st.json(result)  # ← add this temporarily
                    else:
                        st.error("Failed to read prescription. Try a clearer image.")

            # show extracted medicines
            if "prescription_result" in st.session_state and st.session_state.prescription_result:
                result = st.session_state.prescription_result
                st.divider()
                st.subheader("💊 Extracted Medicines")

                # show prescription info
                info_col1, info_col2, info_col3 = st.columns(3)
                info_col1.info(f"👨‍⚕️ Dr: {result.get('doctor_name', 'N/A')}")
                info_col2.info(f"👤 Patient: {result.get('patient_name', 'N/A')}")
                info_col3.info(f"📅 Date: {result.get('date', 'N/A')}")

                medicines = result.get("medicines", [])

                if medicines:
                    # show each extracted medicine
                    for i, med in enumerate(medicines):
                        # build time slots summary
                        slots = []
                        if med.get("morning"): slots.append(f"🌅 Morning: {med['morning']}")
                        if med.get("noon"):    slots.append(f"☀️ Noon: {med['noon']}")
                        if med.get("evening"): slots.append(f"🌆 Evening: {med['evening']}")
                        if med.get("night"):   slots.append(f"🌙 Night: {med['night']}")
                        slots_str = " | ".join(slots) if slots else "Once daily"
                    
                        with st.expander(
                            f"💊 {med.get('name', 'Unknown')} — "
                            f"{med.get('dose', '')} — "
                            f"{slots_str}",
                            expanded=True
                        ):
                            c1, c2, c3 = st.columns(3)
                            c1.write(f"**Name:** {med.get('name', '')}")
                            c2.write(f"**Dose:** {med.get('dose', '')}")
                            c3.write(f"**Route:** {med.get('route', '')}")
                            st.write(f"**Duration:** {med.get('duration', '')}")
                    
                            # show time slots
                            st.markdown("**Schedule:**")
                            if slots:
                                for slot in slots:
                                    st.write(slot)
                            else:
                                st.write("Once daily")
                    
                            if med.get('instructions'):
                                st.warning(f"📝 {med.get('instructions', '')}")
                    st.divider()

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Add All to Medication Tracker",
                                     type="primary",
                                     key="add_all_meds"):
                            added = 0
                            for med in medicines:
                                name         = med.get("name", "Unknown")
                                duration     = med.get("duration", "")
                                route        = med.get("route", "")
                                instructions = med.get("instructions", "")
                                notes        = f"{route} | {instructions}".strip(" |")

                                # build time slots
                                slots = {
                                    "Morning": med.get("morning", ""),
                                    "Noon":    med.get("noon", ""),
                                    "Evening": med.get("evening", ""),
                                    "Night":   med.get("night", "")
                                }

                                # filter only non-empty slots
                                active_slots = {k: v for k, v in slots.items() if v.strip()}

                                if active_slots:
                                    # add one entry per time slot
                                    for time_slot, dose in active_slots.items():
                                        st.session_state.medications.append({
                                            "name":      name,
                                            "dose":      dose,
                                            "frequency": time_slot,  # Morning/Noon/Evening/Night
                                            "times":     [time_slot],
                                            "notes":     notes,
                                            "duration":  duration
                                        })
                                        added += 1
                                else:
                                    # no slot info — add as once daily
                                    st.session_state.medications.append({
                                        "name":      name,
                                        "dose":      med.get("dose", ""),
                                        "frequency": "Once daily",
                                        "times":     [],
                                        "notes":     notes,
                                        "duration":  duration
                                    })
                                    added += 1

                            st.success(f"✅ Added {added} medication entries to tracker!")
                            st.session_state.prescription_result = None
                            st.rerun()

                    with col2:
                        if st.button("🗑️ Clear", key="clear_prescription"):
                            st.session_state.prescription_result = None
                            st.rerun()

        st.divider()
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
                    # show time slot emoji
                    freq = med.get("frequency", "")
                    if freq == "Morning":   slot_emoji = "🌅"
                    elif freq == "Noon":    slot_emoji = "☀️"
                    elif freq == "Evening": slot_emoji = "🌆"
                    elif freq == "Night":   slot_emoji = "🌙"
                    else:                   slot_emoji = "💊"
                    st.write(f"{slot_emoji} **{med['name']}** — {med['dose']}")
                    if med.get("notes"):
                        st.caption(f"📝 {med['notes']}")
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

            # Adherence today — medication specific
            total        = len(st.session_state.medications)
            adherence    = (taken_count / total * 100) if total > 0 else 0
            adherence    = min(adherence, 100)

            st.progress(
                min(adherence/100, 1.0),
                text=f"Today's adherence: {taken_count}/{total} ({adherence:.0f}%)"
            )

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

    # ════════════════════════════════════════════════════════
# EXERCISE TRACKER — upgraded with plan + adherence
# ════════════════════════════════════════════════════════
with tracker_tab3:
    p = st.session_state.get("profile", {})

    if p:
        st.success(
            f"👤 {p.get('name')} | {p.get('gender')} | "
            f"Age: {p.get('age')} | BMI: {p.get('bmi')} | "
            f"{p.get('fitness_level')} | {p.get('diabetes_type')}"
        )

    # initialize session state
    if "exercise_chat"      not in st.session_state:
        st.session_state.exercise_chat      = []
    if "exercise_plan"      not in st.session_state:
        st.session_state.exercise_plan      = []
    if "ex_plan_generated"  not in st.session_state:
        st.session_state.ex_plan_generated  = False
    if "ex_plan_ready"      not in st.session_state:
        st.session_state.ex_plan_ready      = False

    # ── STEP 1: Setup Chat ───────────────────────────────
    if not st.session_state.ex_plan_generated:

        st.markdown("### 💬 Step 1 — Tell me about your fitness")
        st.caption("Quick chat to set up your personalized exercise plan")

        risk_level = st.session_state.get('risk_level', 'Medium Risk')
        risk_prob  = st.session_state.get('risk_prob', 0.5)

        ex_setup_system = f"""You are a certified diabetes fitness coach
helping set up a personalized exercise plan.

Patient Profile:
- Name: {p.get('name', 'Patient')}
- Age: {p.get('age', 'Unknown')}
- Gender: {p.get('gender', 'Unknown')}
- Diabetes Status: {p.get('diabetes_type', 'Unknown')}
- BMI: {p.get('bmi', 'Unknown')}
- Risk Level: {risk_level} ({risk_prob*100:.0f}%)
- Current Fitness: {p.get('fitness_level', 'Sedentary')}
- Known Conditions: {', '.join(p.get('conditions', ['None']))}

Your job:
Ask these 4 questions ONE AT A TIME:
Q1: What type of exercise do you enjoy or prefer?
    (walking, yoga, cycling, swimming, gym, home workout)
Q2: How many days per week can you exercise?
    (suggest 5 days for diabetics — rest 2 days)
Q3: How much time can you spend per session?
    (suggest 30-45 min)
Q4: Any physical limitations or injuries?
    (joint pain, back pain, heart condition, etc.)

After all answers:
- Summarize their preferences
- Give 2-3 tips based on their profile + gender
- End with: "Ready to generate your plan? Type yes!"

Keep responses friendly and concise.
Consider their gender and age for recommendations."""

        # welcome message
        if not st.session_state.exercise_chat:
            with st.chat_message("assistant"):
                welcome = f"""Hi {p.get('name', 'there')}! 💪 
Let's create your personalized 7-day exercise plan.

Based on your profile — **{p.get('gender', '')}, 
Age {p.get('age', '')}, BMI {p.get('bmi', '')}** — 
I'll design a safe and effective plan for diabetes management.

Just 4 quick questions to get started!

**What type of exercise do you enjoy or prefer?**
Some options: walking, yoga, cycling, swimming, 
home workouts, gym. What suits you best?"""
                st.write(welcome)
            st.session_state.exercise_chat.append({
                "role": "assistant", "content": welcome
            })

        # show chat history
        for msg in st.session_state.exercise_chat[1:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # reset button
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Start Over", key="ex_reset"):
                st.session_state.exercise_chat     = []
                st.session_state.exercise_plan     = []
                st.session_state.ex_plan_generated = False
                st.session_state.ex_plan_ready     = False
                st.rerun()

        # chat input
        if user_msg := st.chat_input("Type your answer...",
                                      key="exercise_chat_input"):
            with st.chat_message("user"):
                st.write(user_msg)
            st.session_state.exercise_chat.append({
                "role": "user", "content": user_msg
            })

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = call_groq(
                        system=ex_setup_system,
                        user=user_msg,
                        history=st.session_state.exercise_chat[:-1]
                    )
                st.write(response)
            st.session_state.exercise_chat.append({
                "role": "assistant", "content": response
            })

            if any(word in user_msg.lower() for word in
                   ["yes", "ready", "generate", "ok",
                    "sure", "proceed", "go ahead"]):
                st.session_state.ex_plan_ready = True
                st.rerun()

        # generate button
        if (len(st.session_state.exercise_chat) >= 6
                or st.session_state.ex_plan_ready):
            st.divider()
            st.info("✅ Preferences collected! Ready to generate your plan?")
            if st.button("🏃 Generate My 7-Day Exercise Plan",
                         type="primary",
                         key="generate_ex_plan_btn"):
                st.session_state.ex_plan_generated = True
                st.rerun()

    # ── STEP 2: Generate Plan ────────────────────────────
    elif st.session_state.ex_plan_generated \
            and not st.session_state.exercise_plan:

        p          = st.session_state.get("profile", {})
        risk_level = st.session_state.get('risk_level', 'Medium Risk')

        chat_summary = "\n".join([
            f"{m['role']}: {m['content']}"
            for m in st.session_state.exercise_chat
        ])

        with st.spinner("🧠 Creating your exercise plan..."):

            # extract preferences
            ex_prefs_raw = call_groq(
                system="""Extract exercise preferences from conversation.
Return JSON only — no extra text:
{
  "preferred_exercise": "walking",
  "days_per_week": 5,
  "session_duration": 30,
  "limitations": "none",
  "rest_days": ["Sunday", "Wednesday"]
}""",
                user=f"Extract from:\n{chat_summary}"
            )

            try:
                import json as json_lib
                prefs_clean = ex_prefs_raw.strip()
                if "```json" in prefs_clean:
                    prefs_clean = prefs_clean.split("```json")[1].split("```")[0]
                elif "```" in prefs_clean:
                    prefs_clean = prefs_clean.split("```")[1]
                    if prefs_clean.startswith("json"):
                        prefs_clean = prefs_clean[4:]
                start = prefs_clean.find("{")
                end   = prefs_clean.rfind("}") + 1
                if start != -1 and end > start:
                    prefs_clean = prefs_clean[start:end]
                ex_prefs = json_lib.loads(prefs_clean)
            except:
                ex_prefs = {
                    "preferred_exercise": "walking",
                    "days_per_week":      5,
                    "session_duration":   30,
                    "limitations":        "none",
                    "rest_days":          ["Sunday"]
                }

            # generate 7-day plan
            ex_plan_prompt = f"""Create a 7-day exercise plan for a diabetes patient.

Patient:
- Name: {p.get('name')}
- Gender: {p.get('gender')}
- Age: {p.get('age')}
- BMI: {p.get('bmi')}
- Diabetes: {p.get('diabetes_type')}
- Risk Level: {risk_level}
- Fitness Level: {p.get('fitness_level')}
- Conditions: {', '.join(p.get('conditions', ['None']))}
- Preferred Exercise: {ex_prefs.get('preferred_exercise')}
- Days per week: {ex_prefs.get('days_per_week')}
- Session duration: {ex_prefs.get('session_duration')} minutes
- Limitations: {ex_prefs.get('limitations')}
- Rest days: {', '.join(ex_prefs.get('rest_days', ['Sunday']))}

Return ONLY valid JSON:
{{
  "days": [
    {{
      "day": 1,
      "day_name": "Monday",
      "is_rest_day": false,
      "exercise_type": "Brisk Walking",
      "duration": 30,
      "intensity": "Moderate",
      "calories_est": 150,
      "warm_up": "5 min slow walk",
      "cool_down": "5 min stretching",
      "bs_benefit": "Lowers blood sugar by 20-30 mg/dL",
      "instructions": "Walk at a pace where you can talk but feel slightly breathless"
    }},
    {{
      "day": 2,
      "day_name": "Tuesday",
      "is_rest_day": true,
      "exercise_type": "Rest + Light Stretching",
      "duration": 10,
      "intensity": "Light",
      "calories_est": 30,
      "warm_up": "",
      "cool_down": "",
      "bs_benefit": "Recovery helps muscle glucose uptake",
      "instructions": "Gentle full body stretching for 10 minutes"
    }}
  ]
}}

Important:
- Consider gender ({p.get('gender')}) for intensity
- Include proper rest days
- Vary exercise types across the week
- Keep it safe for diabetes patients
- Be concise — max 2 sentences per field"""

            ex_plan_raw = call_groq(
                system="You are a diabetes fitness coach. Return only valid JSON.",
                user=ex_plan_prompt,
                max_tokens=4000
            )

            # parse plan
            try:
                plan_clean = ex_plan_raw.strip()
                if "```json" in plan_clean:
                    plan_clean = plan_clean.split("```json")[1].split("```")[0]
                elif "```" in plan_clean:
                    plan_clean = plan_clean.split("```")[1]
                    if plan_clean.startswith("json"):
                        plan_clean = plan_clean[4:]
                start = plan_clean.find("{")
                end   = plan_clean.rfind("}") + 1
                if start != -1 and end > start:
                    plan_clean = plan_clean[start:end]
                plan_data = json_lib.loads(plan_clean)
                ex_days   = plan_data.get("days", [])
            except Exception as e:
                st.error(f"Parse error: {e}")
                ex_days = []

            # add tracking fields
            for day in ex_days:
                day["status"]          = "pending"
                day["actual_duration"] = 0
                day["actual_exercise"] = ""
                day["notes"]           = ""

            st.session_state.exercise_plan = ex_days

        if ex_days:
            st.success("✅ Your 7-day exercise plan is ready!")
            st.rerun()
        else:
            st.error("Failed to generate plan. Please try again.")
            st.session_state.ex_plan_generated = False

    # ── STEP 3: Exercise Tracker ─────────────────────────
    elif st.session_state.exercise_plan:
        plan = st.session_state.exercise_plan

        # header
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("### 📅 Your 7-Day Exercise Tracker")
        with col2:
            if st.button("🔄 New Plan", key="new_ex_plan"):
                st.session_state.exercise_chat     = []
                st.session_state.exercise_plan     = []
                st.session_state.ex_plan_generated = False
                st.session_state.ex_plan_ready     = False
                st.rerun()

        # overall stats
        all_days     = plan
        rest_days    = [d for d in all_days if d.get("is_rest_day", False)]
        active_days  = [d for d in all_days if not d.get("is_rest_day", False)]
        done_days    = [d for d in active_days if d["status"] in ["done", "modified"]]
        skipped_days = [d for d in active_days if d["status"] == "skipped"]
        adherence    = (len(done_days) / len(active_days) * 100) if active_days else 0
        adherence    = min(adherence, 100.0)

        # streak counter
        streak     = 0
        done_dates = set()
        for i, day in enumerate(plan):
            if day["status"] in ["done", "modified"]:
                done_dates.add(i)
        for i in range(len(plan)-1, -1, -1):
            if i in done_dates:
                streak += 1
            else:
                break

        # metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("✅ Done",      len(done_days))
        m2.metric("❌ Skipped",   len(skipped_days))
        m3.metric("🛌 Rest Days", len(rest_days))
        m4.metric("🔥 Streak",    f"{streak} days")
        m5.metric("📊 Adherence", f"{adherence:.0f}%")

        st.progress(
            min(adherence/100, 1.0),
            text=f"Weekly adherence: {adherence:.0f}% "
                 f"({len(done_days)}/{len(active_days)} active sessions)"
        )

        st.divider()

        # show each day
        for day_idx, day in enumerate(plan):
            is_rest = day.get("is_rest_day", False)
            status  = day["status"]

            if status == "done":       day_emoji = "🟢"
            elif status == "modified": day_emoji = "🟡"
            elif status == "skipped":  day_emoji = "🔴"
            elif is_rest:              day_emoji = "🛌"
            else:                      day_emoji = "⚪"

            rest_label = " — Rest Day" if is_rest else ""

            with st.expander(
                f"{day_emoji} Day {day['day']} — "
                f"{day['day_name']}{rest_label} | "
                f"{day['exercise_type']} | "
                f"{day['duration']} min | {status.title()}",
                expanded=day_idx == 0
            ):
                col1, col2 = st.columns([3, 1])

                with col1:
                    if is_rest:
                        st.info("🛌 Rest & Recovery Day")
                    st.markdown(f"**Exercise:** {day['exercise_type']}")
                    st.markdown(f"**Duration:** {day['duration']} minutes")
                    st.markdown(f"**Intensity:** {day['intensity']}")
                    st.markdown(f"**Est. Calories:** {day['calories_est']} cal")

                    if day.get("warm_up"):
                        st.caption(f"🔥 Warm up: {day['warm_up']}")
                    if day.get("cool_down"):
                        st.caption(f"❄️ Cool down: {day['cool_down']}")
                    if day.get("instructions"):
                        st.info(f"💡 {day['instructions']}")
                    if day.get("bs_benefit"):
                        st.success(f"🩸 {day['bs_benefit']}")

                with col2:
                    if status == "done":
                        st.success("✅ Done!")
                    elif status == "modified":
                        st.warning("✏️ Modified")
                    elif status == "skipped":
                        st.error("❌ Skipped")
                    else:
                        st.info("⏳ Pending")

                # action buttons
                key_base = f"ex_d{day_idx}"
                b1, b2, b3, b4 = st.columns(4)

                with b1:
                    if st.button("✅ Done",
                                 key=f"exdone_{key_base}",
                                 type="primary" if status == "done"
                                 else "secondary"):
                        plan[day_idx]["status"]          = "done"
                        plan[day_idx]["actual_exercise"] = day["exercise_type"]
                        plan[day_idx]["actual_duration"] = day["duration"]
                        st.session_state.exercise_plan   = plan
                        st.rerun()

                with b2:
                    if st.button("✏️ Modified",
                                 key=f"exmod_{key_base}"):
                        plan[day_idx]["status"]        = "modified"
                        st.session_state.exercise_plan = plan
                        st.rerun()

                with b3:
                    if st.button("❌ Skipped",
                                 key=f"exskip_{key_base}"):
                        plan[day_idx]["status"]        = "skipped"
                        st.session_state.exercise_plan = plan
                        st.rerun()

                with b4:
                    if st.button("↩️ Reset",
                                 key=f"exreset_{key_base}"):
                        plan[day_idx]["status"]          = "pending"
                        plan[day_idx]["actual_exercise"] = ""
                        plan[day_idx]["actual_duration"] = 0
                        plan[day_idx]["notes"]           = ""
                        st.session_state.exercise_plan   = plan
                        st.rerun()

                # modified input
                if status == "modified":
                    col1, col2 = st.columns(2)
                    with col1:
                        actual_ex = st.text_input(
                            "What exercise did you do?",
                            value=day.get("actual_exercise", ""),
                            key=f"actual_ex_{key_base}",
                            placeholder="e.g. Yoga instead of walking"
                        )
                        if actual_ex != day.get("actual_exercise", ""):
                            plan[day_idx]["actual_exercise"] = actual_ex
                            st.session_state.exercise_plan   = plan
                    with col2:
                        actual_dur = st.number_input(
                            "Duration (minutes)",
                            min_value=0, max_value=300,
                            value=day.get("actual_duration", 0),
                            key=f"actual_dur_{key_base}"
                        )
                        if actual_dur != day.get("actual_duration", 0):
                            plan[day_idx]["actual_duration"] = actual_dur
                            st.session_state.exercise_plan   = plan

                    ex_notes = st.text_input(
                        "Notes",
                        value=day.get("notes", ""),
                        key=f"exnotes_{key_base}",
                        placeholder="How did it go?"
                    )
                    if ex_notes != day.get("notes", ""):
                        plan[day_idx]["notes"]         = ex_notes
                        st.session_state.exercise_plan = plan

        # ── Adherence Chart ──────────────────────────────
        st.divider()
        st.subheader("📊 Weekly Exercise Chart")

        day_names   = [d['day_name'][:3] for d in plan]
        durations   = []
        colors_list = []

        for day in plan:
            if day["status"] == "done":
                durations.append(day["duration"])
                colors_list.append("#22c55e")
            elif day["status"] == "modified":
                durations.append(day.get("actual_duration", day["duration"]))
                colors_list.append("#f59e0b")
            elif day["status"] == "skipped":
                durations.append(0)
                colors_list.append("#ef4444")
            elif day.get("is_rest_day"):
                durations.append(day["duration"])
                colors_list.append("#6366f1")
            else:
                durations.append(0)
                colors_list.append("#64748b")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=day_names,
            y=durations,
            marker_color=colors_list,
            name='Duration (min)'
        ))
        fig.add_hline(
            y=30, line_dash="dash",
            line_color="white",
            annotation_text="30 min goal"
        )
        fig.update_layout(
            title="Daily Exercise Duration",
            yaxis_title="Minutes",
            xaxis_title="Day",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "🟢 Done | 🟡 Modified | 🔴 Skipped | "
            "🟣 Rest Day | ⚫ Pending"
        )

        # summary stats
        total_mins = sum(
            d["duration"] if d["status"] == "done"
            else d.get("actual_duration", 0) if d["status"] == "modified"
            else 0
            for d in plan
        )
        total_cals = sum(
            d["calories_est"] for d in plan
            if d["status"] in ["done", "modified"]
        )

        if any(d["status"] != "pending" for d in plan):
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Minutes Exercised", total_mins)
            s2.metric("Calories Burned",         total_cals)
            s3.metric("Sessions Completed",
                      len([d for d in plan
                           if d["status"] in ["done", "modified"]]))

        # download
        st.divider()
        ex_text = f"Exercise Plan\n{'='*40}\n\n"
        for day in plan:
            ex_text += f"\n{day['day_name']}"
            if day.get('is_rest_day'):
                ex_text += " (Rest Day)"
            ex_text += f"\n{'-'*20}\n"
            ex_text += f"Exercise: {day['exercise_type']}\n"
            ex_text += f"Duration: {day['duration']} min\n"
            ex_text += f"Intensity: {day['intensity']}\n"
            ex_text += f"Status: {day['status']}\n"
            if day.get("actual_exercise"):
                ex_text += f"Actually did: {day['actual_exercise']}\n"
            if day.get("notes"):
                ex_text += f"Notes: {day['notes']}\n"

        st.download_button(
            "⬇️ Download Exercise Plan",
            data=ex_text,
            file_name="exercise_plan_tracker.txt",
            mime="text/plain"
        )
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