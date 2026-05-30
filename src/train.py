import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score
import joblib
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import (
    PROCESSED_PATH,
    MODELS_PATH,
    TARGET,
    RANDOM_SEED,
    BAD_RISK,
    DT_PARAM_GRID,
    RF_PARAM_GRID,
    CV_FOLDS,
    SCORING
)

def load_data():
    
    train_df = pd.read_csv(PROCESSED_PATH / "train.csv")
    test_df  = pd.read_csv(PROCESSED_PATH / "test.csv")
    
    print(f"Train loaded: {train_df.shape}")
    print(f"Test loaded:  {test_df.shape}")
    
    #  Separate features and target 
    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    
    X_test  = test_df.drop(columns=[TARGET])
    y_test  = test_df[TARGET]
    
    print(f"\nX_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
    
    print(f"\nTrain target distribution:")
    print(y_train.value_counts())
    
    print(f"\nTest target distribution:")
    print(y_test.value_counts())
    
    return X_train, X_test, y_train, y_test

def cross_validate_models(X_train, y_train):
    
    models = {
        'Decision Tree': ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_SEED)),
            ('model', DecisionTreeClassifier(random_state=RANDOM_SEED))
        ]),
        'Random Forest': ImbPipeline([
            ('smote', SMOTE(random_state=RANDOM_SEED)),
            ('model', RandomForestClassifier(
                random_state=RANDOM_SEED,
                class_weight='balanced'
            ))
        ])
    }
    
    results = {}
    
    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=CV_FOLDS,
            scoring=SCORING,
            n_jobs=-1
        )
        
        results[name] = {
            'mean':   scores.mean().round(4),
            'std':    scores.std().round(4),
            'scores': scores.round(4)
        }
        
        print(f"\n{name}:")
        print(f"  Fold scores:  {scores.round(4)}")
        print(f"  Mean ROC-AUC: {scores.mean():.4f}")
        print(f"  Std Dev:      {scores.std():.4f}")
    
    return results


def tune_models(X_train, y_train):
    
    #  Decision Tree pipeline 
    dt_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_SEED)),
        ('model', DecisionTreeClassifier(random_state=RANDOM_SEED))
    ])
    
    #  Random Forest pipeline 
    rf_pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_SEED)),
        ('model', RandomForestClassifier(
            random_state=RANDOM_SEED,
            class_weight='balanced'
        ))
    ])
    
    #  GridSearchCV for Decision Tree 
    print("\nTuning Decision Tree")
    dt_params = {f'model__{k}': v for k, v in DT_PARAM_GRID.items()}
    dt_search = GridSearchCV(
        dt_pipeline,
        dt_params,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        verbose=1
    )
    dt_search.fit(X_train, y_train)
    
    print(f"Best DT params:    {dt_search.best_params_}")
    print(f"Best DT ROC-AUC:   {dt_search.best_score_:.4f}")
    
    #  GridSearchCV for Random Forest 
    #  GridSearchCV wraps your pipeline and tries every combination of hyperparameters using cross-validation.
    print("\nTuning Random Forest")
    rf_params = {f'model__{k}': v for k, v in RF_PARAM_GRID.items()}
    rf_search = GridSearchCV(
        rf_pipeline,
        rf_params,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        verbose=1
    )
    rf_search.fit(X_train, y_train)
    
    print(f"Best RF params:    {rf_search.best_params_}")
    print(f"Best RF ROC-AUC:   {rf_search.best_score_:.4f}")
    
    return dt_search, rf_search


def train_final_models(X_train, y_train, dt_search, rf_search):
    
    #  Apply SMOTE to full training data 
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print(f"\nTraining data after SMOTE: {X_train_bal.shape}")
    
    #  Extract best parameters 
    dt_params = {
        k.replace('model__', ''): v
        for k, v in dt_search.best_params_.items()
    }
    rf_params = {
        k.replace('model__', ''): v
        for k, v in rf_search.best_params_.items()
    }
    
    #  Train Decision Tree 
    print("\nTraining final Decision Tree...")
    dt_model = DecisionTreeClassifier(
        **dt_params,
        random_state=RANDOM_SEED
    )
    dt_model.fit(X_train_bal, y_train_bal)
    print("Decision Tree trained.")
    
    #  Train Random Forest 
    print("\nTraining final Random Forest...")
    rf_model = RandomForestClassifier(
        **rf_params,
        random_state=RANDOM_SEED,
        class_weight='balanced'
    )
    rf_model.fit(X_train_bal, y_train_bal)
    print("Random Forest trained.")
    
    #  Save models 
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(dt_model, MODELS_PATH / "decision_tree.pkl")
    joblib.dump(rf_model, MODELS_PATH / "random_forest.pkl")
    
    print(f"\nModels saved:")
    print(f"  models/decision_tree.pkl")
    print(f"  models/random_forest.pkl")
    
    #  Quick sanity check 
    dt_train_score = roc_auc_score(
        y_train_bal,
        dt_model.predict_proba(X_train_bal)[:, 1]
    )
    rf_train_score = roc_auc_score(
        y_train_bal,
        rf_model.predict_proba(X_train_bal)[:, 1]
    )
    
    print(f"\nTraining ROC-AUC (sanity check):")
    print(f"  Decision Tree:  {dt_train_score:.4f}")
    print(f"  Random Forest:  {rf_train_score:.4f}")
    print(f"  (these should be high — model has seen this data)")
    
    return dt_model, rf_model



if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    cv_results = cross_validate_models(X_train, y_train)
    dt_search, rf_search = tune_models(X_train, y_train)
    dt_model, rf_model = train_final_models(X_train, y_train,dt_search, rf_search)
                                         




