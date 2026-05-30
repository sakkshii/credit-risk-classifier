import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from config import (
    DATA_PATH,
    PROCESSED_PATH,
    MODELS_PATH,
    TARGET,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    RANDOM_SEED,
    TEST_SIZE,
    BAD_RISK,
    GOOD_RISK,
    ORDINAL_ORDERS,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES
)

def load_and_clean(path):

    df=pd.read_csv(path)
    print(f"raw data loaded: {df.shape}")

    df.columns = df.columns.str.strip()

    # strip whitespaces from string obj
    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols]= df[cat_cols].apply(lambda x: x.str.strip())

    return df

def split_data(df):
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    Y = df[TARGET]

    print(f"\nFeatures shape: {X.shape}")
    print(f"\nTarget shape: {Y.shape}")

    # split 
    X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=Y)

    print(f"\nTrain size: {X_train.shape[0]}")
    print(f"\nTest size: {X_test.shape[0]}")

    # verify class balance 
    print(f"\n Train target")
    print(Y_train.value_counts(normalize=True).round(3)*100)
    print(f"\nTest target distribution")
    print(Y_test.value_counts(normalize=True).round(3)*100)

    return X_train, Y_train, X_test, Y_test

def encode_features(X_train, X_test):

    encoder = ColumnTransformer(
        transformers= [
            (
            'ordinal',
            OrdinalEncoder(categories=ORDINAL_ORDERS, handle_unknown='use_encoded_value', unknown_value=-1),
            ORDINAL_FEATURES 
            ), 
            (
                'onehot',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                NOMINAL_FEATURES
            )
        ], 
        remainder='passthrough'
    )
    #  Fit on train, transform both 
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded  = encoder.transform(X_test)
    
    print(f"\nShape before encoding: {X_train.shape}")
    print(f"Shape after encoding:  {X_train_encoded.shape}")
    
    #  Get feature names 
    onehot_names  = encoder.named_transformers_['onehot']\
                           .get_feature_names_out(NOMINAL_FEATURES)
    feature_names = ORDINAL_FEATURES + list(onehot_names) + NUMERICAL_FEATURES
    
    #  Convert back to DataFrames
    X_train_enc = pd.DataFrame(
        X_train_encoded,
        columns=feature_names
    )
    X_test_enc = pd.DataFrame(
        X_test_encoded,
        columns=feature_names
    )
    
    print(f"\nFinal feature names ({len(feature_names)}):")
    print(feature_names)
    
    return X_train_enc, X_test_enc, encoder


def save_data(X_train_bal, y_train_bal, X_test_enc, y_test, encoder):
    
    #  Create directories if they don't exist 
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    #  Combine features and target back together 
    train_df = pd.DataFrame(X_train_bal)
    train_df[TARGET] = y_train_bal.values if hasattr(y_train_bal, 'values') \
                       else y_train_bal
    
    test_df = pd.DataFrame(X_test_enc)
    test_df[TARGET] = y_test.values
    
    #  Save CSVs 
    train_df.to_csv(PROCESSED_PATH / "train.csv", index=False)
    test_df.to_csv(PROCESSED_PATH  / "test.csv",  index=False)
    
    print(f"\nTrain CSV saved: {train_df.shape}")
    print(f"Test CSV saved:  {test_df.shape}")
    
    #  Save encoder 
    joblib.dump(encoder, MODELS_PATH / "encoder.pkl")
    print(f"\nEncoder saved: models/encoder.pkl")
    
    #  Final verification 
    print(f"\nFinal verification:")
    print(f"Train target distribution:")
    print(train_df[TARGET].value_counts())
    print(f"\nTest target distribution:")
    print(test_df[TARGET].value_counts())



if __name__=="__main__":
    df= load_and_clean(DATA_PATH)
    X_train, Y_train, X_test, Y_test = split_data(df)
    X_train_enc, X_test_enc, encoder = encode_features(X_train, X_test)
    save_data(X_train_enc, Y_train, X_test_enc, Y_test, encoder)
