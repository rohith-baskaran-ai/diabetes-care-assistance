# 🏥 Diabetes Care Assistant

> An AI-powered diabetes management application that helps patients track their health, get personalized guidance, and manage their care — all in one place.

**Live Demo:** https://diabetes-care-assistance-bgqp5eacmz2xdzwleynya3.streamlit.app/

---

## What It Does

Diabetes Care Assistant combines machine learning, large language models, and computer vision to give diabetic patients a complete health companion. Patients can assess their risk, upload prescriptions, track blood sugar and medications, get personalized diet and exercise plans, and chat with an AI health assistant — all personalized to their profile.

---

## Key Features

### 🔍 AI Risk Assessment
Trained a Random Forest classifier on the Pima Indians Diabetes Dataset (768 patients, 8 clinical features). The model predicts diabetes probability with **75.32% accuracy** and surfaces the top contributing risk factors for each patient. After assessment, an LLM generates 3 personalized, actionable recommendations based on the patient's exact metrics.

### 📋 Prescription Upload → Auto Medication Tracker
Upload any photo of a prescription — the app uses **Llama 4 Scout vision model** to extract every medicine, dosage, and time slot automatically. A prescription from Kovai Medical Center with 7 medicines across morning, noon, and night slots was extracted correctly in under 10 seconds. Medicines are added to the tracker with one click.

### 🤖 Agentic Health Tracking
The health assistant detects intent from natural conversation and updates trackers automatically:
- *"My blood sugar is 145 after lunch"* → logs reading to blood sugar tracker
- *"I took my Actrapid morning dose"* → ticks medication as taken
- *"I walked 45 minutes today"* → logs exercise session

No manual navigation. One message does it all.

### 🥗 Personalized Diet Plan with Adherence Tracking
A conversational setup chat collects the patient's preferences — favourite foods, foods to avoid, wake/sleep time, cooking time, weekly goal. The LLM generates a structured 7-day Indian meal plan with multiple items per meal (carbs + calories). Patients mark each meal as Followed / Modified / Skipped, and a weekly adherence chart shows which days they stuck to the plan.

### 🏃 Exercise Plan with Adherence Tracking
Same conversational approach for exercise — considers gender, age, BMI, fitness level, diabetes type, and physical limitations. Generates a 7-day plan with proper rest days, warm-up/cool-down, and blood sugar impact notes. Patients log actual exercise and track streaks.

### 📈 Health Dashboard + PDF Report
A unified dashboard shows today's overview (blood sugar readings, medications taken, exercise, meals logged, risk level), an A1C estimator with gauge chart based on average glucose readings, weekly trend charts, and a downloadable PDF health report suitable for doctor visits.

---

## Full Feature List

| Tab | Features |
|-----|----------|
| 👤 My Profile | Personal info, physical stats, diabetes status, diet preference, blood sugar targets |
| 🔍 Risk Assessment | ML prediction, risk probability, key factors, AI recommendations |
| 💬 Health Assistant | Personalized chatbot, agentic auto-logging, conversation history |
| 🥗 Diet Plan | Chat setup, 7-day structured plan, meal adherence tracker, download |
| 📊 Health Tracker | Blood sugar logs + charts, medication tracker with adherence %, exercise tracker with streaks, food + GI tracker |
| 📈 Dashboard | Today's overview, A1C estimator, weekly trends, PDF health report |

---

## Tech Stack

```
ML Model       Random Forest (scikit-learn)
               Pima Indians Diabetes Dataset
               75.32% accuracy

LLM            Groq API — Llama 3.3 70B Versatile
               Fast inference, low latency

Vision         Meta Llama 4 Scout (17B)
               Prescription OCR and extraction

Frontend       Streamlit
               Plotly for interactive charts

Backend        Python — all logic in app.py

Deployment     Streamlit Cloud (frontend)
               Groq Cloud (LLM inference)
```

---

## Architecture

```
Patient
   ↓
Streamlit UI (6 tabs)
   ↓
┌─────────────────────────────────┐
│  ML Layer        LLM Layer      │
│  Random Forest   Groq API       │
│  Risk prediction Chat + Plans   │
│                  Vision OCR     │
└─────────────────────────────────┘
   ↓
Session State (tracking data)
```

---

## ML Model Details

```
Dataset:     Pima Indians Diabetes (768 patients)
Features:    Glucose, BMI, Age, Blood Pressure,
             Pregnancies, Insulin, Skin Thickness,
             Diabetes Pedigree Function
Model:       Random Forest (100 estimators)
Accuracy:    75.32%
Class weight: Balanced (handles class imbalance)

Feature Importance:
Glucose      ████████████ 0.28
BMI          ████████     0.17
Age          ██████       0.13
DPF          █████        0.12
```

---

## Agentic Behavior

The app demonstrates real agentic AI — the health assistant detects intent from natural language and autonomously updates the relevant tracker without any manual navigation:

```
User message → Intent detection (Llama 3.3)
                      ↓
           ┌──────────┼──────────┐
           ↓          ↓          ↓
    Blood Sugar   Medication  Exercise
      Logger       Tracker     Logger
```

---

## How to Run Locally

```bash
# clone the repo
git clone https://github.com/rohith-baskaran-ai/diabetes-care-assistant
cd diabetes-care-assistant

# install dependencies
pip install -r requirements.txt

# add environment variables
echo "GROQ_API_KEY=your-key" > .env

# train the model
python train_model.py

# run the app
streamlit run app.py
```

---

## Requirements

```
streamlit
scikit-learn
pandas
numpy
joblib
groq
plotly
reportlab
python-dotenv
```

---

## Project Structure

```
diabetes-care-assistant/
├── app.py              ← main Streamlit application
├── train_model.py      ← ML model training script
├── requirements.txt
├── .env                ← API keys (not committed)
└── model/
    ├── diabetes_model.pkl
    ├── scaler.pkl
    └── feature_names.json
```

---

## What I Learned

Building this taught me that the hardest part of AI applications is not the model — it's the product thinking. Getting the LLM to generate structured JSON for a 7-day meal plan, handling token limits by splitting generation into batches, parsing vision model output from real Indian hospital prescriptions, and designing agentic intent detection that works reliably without false positives — these are the real engineering challenges that don't appear in tutorials.

---

## Roadmap

```
Phase 1 (Complete) ✅
→ ML risk assessment
→ LLM health assistant
→ Prescription vision OCR
→ Diet + exercise plan trackers
→ Agentic chat auto-logging
→ Dashboard + PDF reports

Phase 2 (Planned)
→ Supabase database (persistent storage)
→ User authentication (OTP login)
→ WhatsApp integration (Twilio)
→ Doctor dashboard

Phase 3 (Future)
→ Flutter mobile app
→ Push notifications
→ Tamil language support
→ Glucometer integration
```

---

## About

Built as part of a structured AI Developer learning roadmap covering Python, FastAPI, LLM APIs, RAG, LangGraph agents, and full-stack deployment. This project applies all those skills to a real-world health problem affecting 77 million people in India.

---

*Always consult a qualified doctor for medical decisions. This app is for tracking and educational purposes only.*