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

    import warnings
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    import numpy as np

    # ── Ordinal encoding ──────────────────────────────
    ordinal_mapping = {
        'status': {
            'no checking account': 0,
            '... < 100 DM': 1,
            '0 <= ... < 200 DM': 2,
            '... >= 200 DM / salary assignments for at least 1 year': 3
        },
        'savings': {
            'unknown/no savings account': 0,
            '... < 100 DM': 1,
            '100 <= ... < 500 DM': 2,
            '500 <= ... < 1000 DM': 3,
            '... >= 1000 DM': 4
        },
        'employment_duration': {
            'unemployed': 0,
            '... < 1 year': 1,
            '1 <= ... < 4 years': 2,
            '4 <= ... < 7 years': 3,
            '... >= 7 years': 4
        },
        'credit_history': {
            'no credits taken/all credits paid back duly': 0,
            'all credits at this bank paid back duly': 1,
            'existing credits paid back duly till now': 2,
            'delay in paying off in the past': 3,
            'critical account/other credits existing': 4
        }
    }

    # ── One-hot categories ────────────────────────────
    onehot_categories = {
        'purpose': ['business', 'car (new)', 'car (used)',
                    'domestic appliances', 'education',
                    'furniture/equipment', 'others',
                    'radio/television', 'repairs', 'retraining'],
        'personal_status_sex': [
            'female : divorced/separated/married',
            'male : divorced/separated',
            'male : married/widowed',
            'male : single'
        ],
        'other_debtors': ['co-applicant', 'guarantor', 'none'],
        'property': [
            'building society savings agreement/life insurance',
            'car or other', 'real estate', 'unknown/no property'
        ],
        'other_installment_plans': ['bank', 'none', 'stores'],
        'housing': ['for free', 'own', 'rent'],
        'job': [
            'management/self-employed/highly qualified employee/officer',
            'skilled employee/official',
            'unemployed/unskilled - non-resident',
            'unskilled - resident'
        ],
        'telephone': ['no', 'yes'],
        'foreign_worker': ['no', 'yes']
    }

    # ── Build feature vector ──────────────────────────
    features = []

    # Ordinal features
    for feat in ORDINAL_FEATURES:
        features.append(ordinal_mapping[feat][applicant_dict[feat]])

    # One-hot features
    for feat in NOMINAL_FEATURES:
        categories = onehot_categories[feat]
        for cat in categories:
            features.append(1 if applicant_dict[feat] == cat else 0)

    # Numerical features
    for feat in NUMERICAL_FEATURES:
        features.append(applicant_dict[feat])

    # ── Predict ───────────────────────────────────────
    input_array = np.array(features).reshape(1, -1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction  = rf_model.predict(input_array)[0]
        probability = rf_model.predict_proba(input_array)[0]

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





