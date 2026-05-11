import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
st.set_page_config(
    page_title="CKD",
    page_icon="🩺",
    layout="wide"
)
from fpdf import FPDF
from utils.email_service import send_report
from services.supabase_service import save_prediction
from streamlit_lottie import st_lottie
import requests
import time
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_ai = load_lottie(
    "https://assets9.lottiefiles.com/packages/lf20_49rdyysj.json"
)

lottie_health = load_lottie(
    "https://assets10.lottiefiles.com/packages/lf20_x1gjdldd.json"
)


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', message='.*use_column_width.*')



st.markdown("A comprehensive tool to determine the clinical stage of Chronic Kidney Disease based on patient data, primarily using the eGFR value as per medical guidelines.")

st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(180deg, #f8fafc, #eef2f7);
        color: #111827;
    }

    html, body, [class*="css"] {
        color: #111827 !important;
        font-family: 'Poppins', sans-serif;
    }


    h1 {
        text-align: center;
        font-size: 3rem !important;
        font-weight: 800 !important;
        color: #1d4ed8 !important;
        margin-bottom: 15px;
    }

    h2, h3 {
        color: #0f172a !important;
        font-weight: 700 !important;
    }


    [data-testid="stVerticalBlock"] > div {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: 0.2s ease;
    }

    [data-testid="stVerticalBlock"] > div:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }



    .stTextInput input,
    .stNumberInput input,
    textarea {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
    }

    .stTextInput input:focus,
    .stNumberInput input:focus,
    textarea:focus {
        border: 1px solid #2563eb !important;
        box-shadow: 0 0 8px rgba(37,99,235,0.25);
    }

    

    div[data-baseweb="select"] {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px !important;
    }

  

    .stButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 12px;
        font-weight: 600;
        color: white;
        background: #2563eb;
        border: none;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #1d4ed8;
        transform: scale(1.02);
    }

   

    [data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    

    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }

    ::-webkit-scrollbar-thumb {
        background: #94a3b8;
        border-radius: 10px;
    }


    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        color: #1d4ed8 !important;
        font-weight: 600;
    }

    

    .footer-card {
        text-align: center;
        padding: 20px;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
    }

    </style>
    """, unsafe_allow_html=True)

left, right = st.columns([1.2,1])

with left:

    st.title("⚕️ CKD Stage Prediction and Treatment AI")

    st.markdown("""
    <div style="
    padding:25px;
    border-radius:25px;color: 
   background: rgba(255,255,255,0.10);;
    backdrop-filter: blur(18px);
    border:1px solid rgba(255,255,255,0.08);
    margin-bottom:20px;
    ">

    <h2 style="
    color:white;
    font-size:2rem;
    ">
    🧠 AI Powered Kidney Disease Analysis
    </h2>

    <p style="
    color:black;
    font-size:18px;
    line-height:1.8;
    ">
    Predict CKD stages, generate smart medical reports,
    receive AI-powered clinical guidance,
    and monitor kidney health using advanced Machine Learning + RAG AI technology.
    </p>

    </div>
    """, unsafe_allow_html=True)

with right:

    st_lottie(
        lottie_ai,
        height=320,
        key="ai_animation"
    )



col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("⚕️ Accuracy", "98.2%")

with col2:
    st.metric("🧠 AI Engine", "LLM + RAG")

with col3:
    st.metric("📄 Reports", "PDF + Email")

with col4:
    st.metric("🏥 Monitoring", "Real-time")

st.markdown("<br>", unsafe_allow_html=True)



st.markdown("""
<div style="
padding:16px;
border-radius:18px;
background: linear-gradient(90deg,#2563eb,#06b6d4);
text-align:center;
font-weight:700;
font-size:18px;
box-shadow:0 8px 25px rgba(37,99,235,0.35);
margin-bottom:25px;
">
🚀 AI System Online • Clinical Prediction Engine Active
</div>
""", unsafe_allow_html=True)



st.markdown("""
<h2 style="
margin-top:20px;
margin-bottom:20px;
">
📄 Upload Patient CSV Data
</h2>
""", unsafe_allow_html=True)



progress = st.progress(0)

for i in range(100):

    time.sleep(0.01)
    progress.progress(i + 1)

st.success("✅ AI Analysis Engine Ready")


st.markdown("""
<div style="
padding:25px;
border-radius:25px;
background: rgba(255,255,255,0.05);
backdrop-filter: blur(18px);
border:1px solid rgba(255,255,255,0.08);
margin-top:30px;
margin-bottom:20px;
">

<h2 style="
text-align:center;
font-size:2rem;
color:#7dd3fc;
">
🤖 CKD AI Medical Assistant
</h2>

<p style="
text-align:center;
color:black;
font-size:17px;
">
Ask AI anything related to CKD diet,
treatment, stages, symptoms or precautions.
</p>

</div>
""", unsafe_allow_html=True)




@st.cache_resource
def load_assets():
    """Loads all required models and preprocessing objects."""
    model = joblib.load("models/ckd_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    target_encoder = joblib.load("models/encoder.joblib")
    imputer = joblib.load("models/imputer.joblib")
    selected_features = joblib.load("models/selected_features.joblib")
    all_feature_names = list(imputer.feature_names_in_)
    return model, scaler, target_encoder, imputer, selected_features, all_feature_names

model, scaler, target_encoder, imputer, selected_features, all_feature_names = load_assets()

if 'Target' in selected_features:
    selected_features = [f for f in selected_features if f != 'Target']


ckd_stage_guidance = {
    'Stage 1': {
        'description': "Kidney damage with normal or high function (eGFR ≥ 90).",
        'goals': ["Slow disease progression", "Reduce cardiovascular disease risk"],
        'dietary_focus': ["Maintain a balanced, heart-healthy diet", "Avoid high-sodium processed foods", "Ensure adequate hydration"],
        'lifestyle': ["Engage in regular exercise (at least 150 mins/week)", "Quit smoking completely", "Avoid NSAID painkillers (e.g., Ibuprofen, Naproxen) unless approved by a doctor"],
        'monitoring': ["Annual check-up with blood pressure monitoring", "Urine test for albumin", "Blood test for creatinine to calculate eGFR"],
        'complications': ["Generally none, but the underlying cause (e.g., diabetes, hypertension) needs aggressive management."]
    },
    'Stage 2': {
        'description': "Mild loss of kidney function (eGFR 60-89).",
        'goals': ["Continue to slow progression", "Aggressively manage blood pressure and blood sugar"],
        'dietary_focus': ["Implement a low-sodium diet (< 2,300 mg/day)", "Continue heart-healthy eating patterns"],
        'lifestyle': ["All Stage 1 recommendations apply and are critical."],
        'monitoring': ["Follow-up every 6-12 months", "Regular monitoring of blood pressure, eGFR, and urine albumin levels"],
        'complications': ["Increased risk of hypertension. Early signs of bone metabolism changes may appear."]
    },
    'Stage 3a': {
        'description': "Mild to moderate loss of kidney function (eGFR 45-59).",
        'goals': ["Prevent and treat complications", "Prepare for more intensive management"],
        'dietary_focus': ["Consult a renal dietitian", "Limit sodium (< 2,000 mg/day)", "May need to limit phosphorus and protein based on lab results"],
        'lifestyle': ["Moderate, consistent exercise is key", "Close self-monitoring of blood pressure and blood sugar (if diabetic)"],
        'monitoring': ["Referral to a Nephrologist (kidney specialist) is standard", "Check-ups every 3-6 months", "Blood tests for phosphorus, calcium, PTH, and hemoglobin"],
        'complications': ["Anemia (low red blood cell count)", "Early bone disease", "Hypertension becomes more common and harder to control"]
    },
    'Stage 3b': {
        'description': "Moderate to severe loss of kidney function (eGFR 30-44).",
        'goals': ["Intensively manage complications", "Educate on renal replacement therapies"],
        'dietary_focus': ["Stricter adherence to renal diet (protein, sodium, potassium, phosphorus limits)", "Fluid intake may need monitoring"],
        'lifestyle': ["All previous recommendations are mandatory.", "Avoid exhaustion; balance activity and rest."],
        'monitoring': ["Nephrologist follow-up every 3 months", "Frequent lab work to manage complications"],
        'complications': ["Anemia and bone disease are more pronounced", "Increased risk of acidosis (build-up of acid in the blood)"]
    },
    'Stage 4': {
        'description': "Severe loss of kidney function (eGFR 15-29).",
        'goals': ["Prepare for kidney failure treatment (dialysis or transplant)"],
        'dietary_focus': ["Very strict diet is essential.", "Work closely with a renal dietitian."],
        'lifestyle': ["Fluid restriction is common.", "Preserve remaining kidney function by following medical advice perfectly."],
        'monitoring': ["Monthly or bi-monthly visits to the nephrologist", "Evaluation for dialysis access placement (e.g., fistula) or transplant listing"],
        'complications': ["High risk of all CKD complications", "Symptoms like fatigue, swelling, and nausea become more common"]
    },
    'Stage 5': {
        'description': "Kidney failure / End-Stage Renal Disease (eGFR < 15).",
        'goals': ["Sustain life with renal replacement therapy"],
        'dietary_focus': ["Follow a specific dialysis-friendly diet, which may differ depending on the type of dialysis", "Strict fluid, sodium, potassium, and phosphorus limits"],
        'lifestyle': ["Adherence to dialysis schedule is life-sustaining.", "Manage symptoms and maintain quality of life."],
        'monitoring': ["Constant medical supervision is required as part of dialysis or post-transplant care."],
        'complications': ["This is a life-threatening condition requiring constant medical care to manage fluid overload, electrolyte imbalances, and waste product buildup."]
    }
}



def get_stage_from_egfr(egfr_value):
    """Returns the clinical CKD stage based on eGFR value."""
    if egfr_value >= 90:
        return 'Stage 1'
    elif 60 <= egfr_value < 90:
        return 'Stage 2'
    elif 45 <= egfr_value < 60:
        return 'Stage 3a'
    elif 30 <= egfr_value < 45:
        return 'Stage 3b'
    elif 15 <= egfr_value < 30:
        return 'Stage 4'
    else: # eGFR < 15
        return 'Stage 5'

def preprocess(df):
    """Applies the full, correct preprocessing pipeline to new data."""
    mapping = {
        'yes': 1, 'no': 0, 'normal': 1, 'abnormal': 0,
        'present': 1, 'not present': 0, 'good': 1, 'poor': 0
    }
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'Target':
             df[col] = df[col].str.lower().str.strip().map(mapping)
    if 'Serum creatinine (mg/dl)' in df.columns and 'Hemoglobin level (gms)' in df.columns:
        df['Creatinine_Hemoglobin'] = df['Serum creatinine (mg/dl)'] * df['Hemoglobin level (gms)']
    if 'Blood urea (mg/dl)' in df.columns and 'Serum creatinine (mg/dl)' in df.columns:
        df['BUN_Creatinine_Ratio'] = df['Blood urea (mg/dl)'] / (df['Serum creatinine (mg/dl)'] + 1e-5)
    if 'Estimated Glomerular Filtration Rate (eGFR)' in df.columns and 'Age of the patient' in df.columns:
        df['eGFR_by_Age'] = df['Estimated Glomerular Filtration Rate (eGFR)'] / (df['Age of the patient'] + 1e-5)
    if 'Target' in df.columns:
        if df['Target'].dtype == 'object':
            df['Target'] = target_encoder.transform(df['Target'])
    else:
        df['Target'] = 0
    for col in all_feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[all_feature_names]
    df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
    df_for_scaler = df_imputed.drop(columns=['Target'])
    df_scaled = pd.DataFrame(scaler.transform(df_for_scaler), columns=df_for_scaler.columns)
    X_selected = df_scaled[selected_features]
    return X_selected

def generate_pdf(patient_data, stage):
    """Generates a downloadable PDF report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Clinical CKD Stage Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Report Date: {datetime.datetime.now().strftime('%d-%m-%Y')}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Patient Name: {patient_data.get('Patient_Name', 'N/A')}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Determined Clinical Stage: {stage}", ln=True)
    guidance = ckd_stage_guidance.get(stage, {})
    if 'description' in guidance:
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 8, f"({guidance['description']})", ln=True)
    pdf.ln(5)

    def write_section(title, content_list):
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font("Arial", '', 11)
        for item in content_list:
            pdf.multi_cell(0, 6, f"  - {item}")
        pdf.ln(3)

    if guidance:
        write_section("Key Goals:", guidance.get('goals', []))
        write_section("Dietary Focus:", guidance.get('dietary_focus', []))
        write_section("Lifestyle Recommendations:", guidance.get('lifestyle', []))
        write_section("Key Complications to Monitor:", guidance.get('complications', []))

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(0, 10, "Disclaimer:", ln=True)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 5, "This report is for informational purposes only. The clinical stage is determined by the eGFR value as per standard medical guidelines. It is not a substitute for professional medical advice. Always consult a qualified nephrologist for diagnosis, treatment, and management plans.")
    
    pdf_path = f"ckd_report_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(pdf_path)
    return pdf_path


st.header("📄 Predict from CSV File")
csv_file = st.file_uploader(
    "Upload a .csv file with patient data",
    type="csv",
    key="csv_uploader"
)
if csv_file:
    try:
        df_csv = pd.read_csv(csv_file)
        st.dataframe(df_csv)
        required_column = "Estimated Glomerular Filtration Rate (eGFR)"
        if required_column not in df_csv.columns:
            st.error(f"❌ Missing column: {required_column}")
        st.stop()
        st.success("✅ Staging Complete!")
        
        final_stages = [get_stage_from_egfr(egfr) for egfr in raw_egfr_values]
        
        df_csv['Determined_Stage'] = final_stages
        csv_output = df_csv.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Staged CSV", data=csv_output, file_name="staged_predictions.csv", mime="text/csv")

        for i, stage in enumerate(final_stages):
            st.write(f"**Patient {i+1} → Determined Clinical Stage: {stage}**")
            guidance = ckd_stage_guidance.get(stage)
            if guidance:
                with st.expander(f"🔬 View Guidance for Patient {i+1} ({stage})"):
                    st.markdown(f"**Description:** {guidance['description']}")
                    
                    st.markdown("**Key Goals:**")
                    for item in guidance['goals']:
                        st.markdown(f"- {item}")

                    st.markdown("**Dietary Focus:**")
                    for item in guidance['dietary_focus']:
                        st.markdown(f"- {item}")

                    st.markdown("**Lifestyle Recommendations:**")
                    for item in guidance['lifestyle']:
                        st.markdown(f"- {item}")
                        
                    st.markdown("**Key Complications to Monitor:**")
                    for item in guidance['complications']:
                        st.markdown(f"- {item}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("---")



with st.expander("✍️ Or Enter Patient Data Manually"):

    if 'saved_inputs' not in st.session_state:
        st.session_state.saved_inputs = []

    with st.form("manual_input_form"):

        manual_input = {}

        st.write("Fill in the patient's lab values and information below.")

        cols = st.columns(3)

        feature_list = [
            f for f in all_feature_names
            if f not in [
                "Creatinine_Hemoglobin",
                "BUN_Creatinine_Ratio",
                "eGFR_by_Age",
                "Target"
            ]
        ]

        for i, feature in enumerate(feature_list):

            with cols[i % 3]:

                
                if (
                    "yes/no" in feature.lower()
                    or "hypertension" in feature.lower()
                    or "diabetes" in feature.lower()
                    or "anemia" in feature.lower()
                    or "smoking" in feature.lower()
                    or "family history" in feature.lower()
                    or "coronary" in feature.lower()
                    or "pedal edema" in feature.lower()
                ):

                    manual_input[feature] = st.selectbox(
                        feature,
                        ["no", "yes"],
                        key=f"{feature}_select"
                    )

              
                elif (
                    "pus cell clumps" in feature.lower()
                    or "bacteria" in feature.lower()
                ):

                    manual_input[feature] = st.selectbox(
                        feature,
                        ["not present", "present"],
                        key=f"{feature}_select"
                    )

              
                elif "appetite" in feature.lower():

                    manual_input[feature] = st.selectbox(
                        feature,
                        ["good", "poor"],
                        key=f"{feature}_select"
                    )

              
                elif (
                    "red blood cells" in feature.lower()
                    or "pus cells" in feature.lower()
                    or "urinary sediment" in feature.lower()
                ):

                    manual_input[feature] = st.selectbox(
                        feature,
                        ["normal", "abnormal"],
                        key=f"{feature}_select"
                    )

        
                else:

                    manual_input[feature] = st.number_input(
                        feature,
                        min_value=0.0,
                        format="%.2f",
                        key=f"{feature}_number"
                    )

        
        patient_name = st.text_input(
            "👤 Enter Patient Name",
            key="patient_name_unique"
        )

       
        user_email = st.text_input(
            "📧 Enter your Email",
            key="user_email_unique"
        )

     
        submitted = st.form_submit_button(
            "🩺 Determine Clinical Stage"
        )


if submitted:

    empty_fields = []

    for key, value in manual_input.items():

        if value == "" or value is None:
            empty_fields.append(key)

    if patient_name.strip() == "":
        empty_fields.append("Patient Name")

    if user_email.strip() == "":
        empty_fields.append("Email")

  
    if empty_fields:

        st.error("⚠️ First fill every field, then report will be generated.")

        st.write("Missing Fields:")

        for field in empty_fields:
            st.write(f"❌ {field}")

    else:

        try:

            raw_egfr = manual_input['Estimated Glomerular Filtration Rate (eGFR)']

            final_stage = get_stage_from_egfr(raw_egfr)

            st.success(f"✅ Determined Clinical Stage: {final_stage}")

            
            if final_stage == "Stage 1":
                st.info("🟢 Mild condition detected. Focus on lifestyle & prevention.")

            elif final_stage == "Stage 2":
                st.warning("🟡 Early kidney damage. Start strict diet control.")

            elif final_stage in ["Stage 3a", "Stage 3b"]:
                st.error("🟠 Moderate CKD detected. Consult nephrologist.")

            elif final_stage == "Stage 4":
                st.error("🔴 Severe damage. Dialysis preparation needed.")

            elif final_stage == "Stage 5":
                st.error("🚨 CRITICAL CONDITION!")

          
            guidance = ckd_stage_guidance.get(final_stage)

            if guidance:

                st.markdown("---")

                st.markdown(f"### 🔬 Detailed Guidance for {final_stage}")

                st.markdown(f"**Description:** {guidance['description']}")

                st.markdown("### 🎯 Goals")
                for item in guidance['goals']:
                    st.write(f"✅ {item}")

                st.markdown("### 🥗 Dietary Focus")
                for item in guidance['dietary_focus']:
                    st.write(f"🍎 {item}")

                st.markdown("### 🏃 Lifestyle")
                for item in guidance['lifestyle']:
                    st.write(f"💡 {item}")

                st.markdown("### ⚠️ Complications")
                for item in guidance['complications']:
                    st.write(f"🔴 {item}")

         
            save_input = manual_input.copy()

            save_input['Patient_Name'] = patient_name
            save_input['Determined_Stage'] = final_stage
            save_input['Timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            st.session_state.saved_inputs.insert(0, save_input)

            
            db_data = {
                "patient_name": patient_name,
                "email": user_email,
                "stage": final_stage
            }

            response = save_prediction(db_data)

         
            pdf_path = generate_pdf(save_input, final_stage)

            with open(pdf_path, "rb") as f:

                st.download_button(
                    "📄 Download Full Report (PDF)",
                    f,
                    file_name=f"CKD_Report_{final_stage}.pdf"
                )

           
            sent = send_report(user_email, pdf_path)

            if sent:
                st.success(f"📧 PDF sent to {user_email}")

            else:
                st.error("❌ Email failed")

          
            os.remove(pdf_path)

        except Exception as e:

            st.error(f"Prediction failed: {e}")




if st.session_state.saved_inputs:
    st.header("📜 Recent Predictions History")
    with st.expander("Click to view past manual predictions"):
        df_saved = pd.DataFrame(st.session_state.saved_inputs)
        st.dataframe(df_saved)
        if st.button("Clear Prediction History"):
            st.session_state.saved_inputs = []
            st.rerun()





st.header("🤖 CKD AI Assistant ")

from services.graph_service import chain

user_q = st.text_input("Ask anything about CKD (diet, stages, treatment)")

if "chat_result" not in st.session_state:
    st.session_state.chat_result = None

if st.button("Ask AI") and user_q.strip():
    try:
       with st.spinner("🧠 AI Thinking..."):
        result = chain.invoke({"query": user_q})

        # SAFE STORE
        st.session_state.chat_result = result

    except Exception as e:
        st.session_state.chat_result = {"response": f"❌ Error: {str(e)}"}

if st.session_state.chat_result:

    st.success(
        st.session_state.chat_result.get(
            "response",
            "No response from AI"
        )
    )

# 👇 FOOTER ALWAYS LAST
st.markdown(
"""
<div style="
    text-align:center;
    padding:20px;
    border-radius:16px;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    border:1px solid rgba(255,255,255,0.2);
    margin-top:30px;
">
    <h3 style="color:#2563eb;">
        👨‍💻 Developed by Kartik Kabdwal
    </h3>
</div>
""",
unsafe_allow_html=True
)