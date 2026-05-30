import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.predict import load_model_and_encoder, predict_risk
from config import (
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
    NUMERICAL_FEATURES
)

st.set_page_config(
    page_title="Credit Risk Classifier",
    page_icon="💳",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp {
        background-color: white;
        color: black;
    }
    .stSlider label, .stSelectbox label, 
    .stNumberInput label, .stSubheader,
    p, h1, h2, h3, label, span {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return load_model_and_encoder()

rf_model, encoder = load_model()

st.title("💳 Credit Risk Classifier")
st.markdown("Enter applicant details to assess credit risk.")
st.divider()

# ── Input form ────────────────────────────────────
st.subheader("📋 Loan Details")
col1, col2 = st.columns(2)

with col1:
    duration = st.slider("Loan duration (months)", 4, 72, 24)
    amount   = st.number_input("Loan amount (DM)", 250, 18424, 3000)
    age      = st.slider("Applicant age", 19, 75, 35)

with col2:
    installment_rate  = st.selectbox(
        "Installment rate (% of income)", [1, 2, 3, 4])
    present_residence = st.selectbox(
        "Years at current residence", [1, 2, 3, 4])
    number_credits    = st.selectbox(
        "Number of existing credits", [1, 2, 3, 4])

people_liable = st.selectbox("Number of financial dependants", [1, 2])

st.divider()
st.subheader("💰 Financial Profile")
col3, col4 = st.columns(2)

with col3:
    status = st.selectbox("Checking account balance", [
        'no checking account',
        '... < 100 DM',
        '0 <= ... < 200 DM',
        '... >= 200 DM / salary assignments for at least 1 year'
    ])
    savings = st.selectbox("Savings account balance", [
        'unknown/no savings account',
        '... < 100 DM',
        '100 <= ... < 500 DM',
        '500 <= ... < 1000 DM',
        '... >= 1000 DM'
    ])

with col4:
    credit_history = st.selectbox("Credit history", [
        'no credits taken/all credits paid back duly',
        'all credits at this bank paid back duly',
        'existing credits paid back duly till now',
        'delay in paying off in the past',
        'critical account/other credits existing'
    ])
    employment_duration = st.selectbox("Employment duration", [
        'unemployed',
        '... < 1 year',
        '1 <= ... < 4 years',
        '4 <= ... < 7 years',
        '... >= 7 years'
    ])

st.divider()
st.subheader("👤 Personal Profile")
col5, col6 = st.columns(2)

with col5:
    purpose = st.selectbox("Purpose of loan", [
        'car (new)', 'car (used)', 'furniture/equipment',
        'radio/television', 'domestic appliances', 'repairs',
        'education', 'retraining', 'business', 'others'
    ])
    housing = st.selectbox("Housing situation", [
        'own', 'rent', 'for free'
    ])
    job = st.selectbox("Employment type", [
        'skilled employee/official',
        'unskilled - resident',
        'management/self-employed/highly qualified employee/officer',
        'unemployed/unskilled - non-resident'
    ])

with col6:
    personal_status_sex = st.selectbox("Personal status", [
        'male : single',
        'male : married/widowed',
        'male : divorced/separated',
        'female : divorced/separated/married'
    ])
    other_debtors = st.selectbox("Other debtors / guarantors", [
        'none', 'co-applicant', 'guarantor'
    ])
    property = st.selectbox("Most valuable property", [
        'real estate',
        'building society savings agreement/life insurance',
        'car or other',
        'unknown/no property'
    ])

col7, col8 = st.columns(2)
with col7:
    other_installment_plans = st.selectbox(
        "Other installment plans", ['none', 'bank', 'stores'])
with col8:
    telephone = st.selectbox(
        "Telephone registered", ['yes', 'no'])

foreign_worker = st.selectbox("Foreign worker", ['yes', 'no'])


# ── Predict button ────────────────────────────────
st.divider()

if st.button("Assess Credit Risk", type="primary", use_container_width=True):
    
    # ── Build applicant dict ──────────────────────
    applicant = {
        'status':                   status,
        'credit_history':           credit_history,
        'purpose':                  purpose,
        'savings':                  savings,
        'employment_duration':      employment_duration,
        'personal_status_sex':      personal_status_sex,
        'other_debtors':            other_debtors,
        'property':                 property,
        'other_installment_plans':  other_installment_plans,
        'housing':                  housing,
        'job':                      job,
        'telephone':                telephone,
        'foreign_worker':           foreign_worker,
        'duration':                 duration,
        'amount':                   amount,
        'installment_rate':         installment_rate,
        'present_residence':        present_residence,
        'age':                      age,
        'number_credits':           number_credits,
        'people_liable':            people_liable
    }
    
    # ── Run prediction ────────────────────────────
    result = predict_risk(applicant, rf_model, encoder)
    
    # ── Display result ────────────────────────────
    st.divider()
    
    if result['label'] == 'Good Risk':
        st.success(f" {result['label']}")
    else:
        st.error(f"{result['label']}")
    
    # ── Probability breakdown ─────────────────────
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(
            label="Good Risk Probability",
            value=f"{result['good_risk_prob']*100:.1f}%"
        )
    with col_b:
        st.metric(
            label="Bad Risk Probability",
            value=f"{result['bad_risk_prob']*100:.1f}%"
        )