import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from config import PROCESSED_PATH, DATA_PATH

import warnings
warnings.filterwarnings("ignore")

from config import (
    MODELS_PATH,
    TARGET,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
    RANDOM_SEED,
    BAD_RISK,
    GOOD_RISK
)

def load_model_and_encoder():
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from config import ORDINAL_FEATURES, ORDINAL_ORDERS, NOMINAL_FEATURES

    rf_model = joblib.load(MODELS_PATH / "random_forest.pkl")

    print(f"Model loaded:   {type(rf_model).__name__}")

    # Rebuild encoder from scratch — avoids pkl version issues
    encoder = ColumnTransformer(
        transformers=[
            (
                'ordinal',
                OrdinalEncoder(
                    categories=ORDINAL_ORDERS,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                ),
                ORDINAL_FEATURES
            ),
            (
                'onehot',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False
                ),
                NOMINAL_FEATURES
            )
        ],
        remainder='passthrough'
    )

    # Fit on the processed training data
    train_df = pd.read_csv(PROCESSED_PATH / "train.csv")
    
    # Get original unencoded columns from raw data
    raw_df = pd.read_csv(DATA_PATH)
    from config import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET
    X_raw = raw_df[ORDINAL_FEATURES + NOMINAL_FEATURES + NUMERICAL_FEATURES]
    encoder.fit(X_raw)

    print(f"Encoder rebuilt and fitted.")
    return rf_model, encoder


def predict_risk(applicant_dict, rf_model, encoder):
    
    #  Convert input to DataFrame 
    input_df = pd.DataFrame([applicant_dict])
    
    #  Correct column order 
    feature_cols  = ORDINAL_FEATURES + NOMINAL_FEATURES + NUMERICAL_FEATURES
    input_df      = input_df[feature_cols]
    
    #  Encode 
    input_encoded = encoder.transform(input_df)
    
    #  Convert to numpy to avoid name mismatch 
    input_array   = np.array(input_encoded)
    
    #  Predict 
    prediction    = rf_model.predict(input_array)[0]
    probability   = rf_model.predict_proba(input_array)[0]
    
    # ── Result 
    label = "Good Risk" if prediction == GOOD_RISK else "Bad Risk"
    
    return {
        'label':          label,
        'prediction':     int(prediction),
        'bad_risk_prob':  round(float(probability[0]), 4),
        'good_risk_prob': round(float(probability[1]), 4)
    }

if __name__ == "__main__":
    rf_model, encoder = load_model_and_encoder()
    
    # ── Test with a sample applicant 
    test_applicant = {
        'status':                   'no checking account',
        'credit_history':           'existing credits paid back duly till now',
        'purpose':                  'car (new)',
        'savings':                  '... < 100 DM',
        'employment_duration':      '1 <= ... < 4 years',
        'personal_status_sex':      'male : single',
        'other_debtors':            'none',
        'property':                 'real estate',
        'other_installment_plans':  'none',
        'housing':                  'own',
        'job':                      'skilled employee/official',
        'telephone':                'no',
        'foreign_worker':           'yes',
        'duration':                 24,
        'amount':                   3000,
        'installment_rate':         2,
        'present_residence':        2,
        'age':                      35,
        'number_credits':           1,
        'people_liable':            1
    }
    
    result = predict_risk(test_applicant, rf_model, encoder)
    
    print(f"\nPrediction:      {result['label']}")
    print(f"Bad Risk Prob:   {result['bad_risk_prob']}")
    print(f"Good Risk Prob:  {result['good_risk_prob']}")





