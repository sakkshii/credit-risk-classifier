import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_curve
)

from config import (
    PROCESSED_PATH,
    MODELS_PATH,
    FIGURES_PATH,
    TARGET,
    RANDOM_SEED,
    BAD_RISK,
    GOOD_RISK,
    COST_FN,
    COST_FP
)

def load_models_and_data():
    
    # ── Load models ───────────────────────────────────
    dt_model = joblib.load(MODELS_PATH / "decision_tree.pkl")
    rf_model = joblib.load(MODELS_PATH / "random_forest.pkl")
    
    print("Models loaded:")
    print(f"  Decision Tree:  {type(dt_model).__name__}")
    print(f"  Random Forest:  {type(rf_model).__name__}")
    
    # ── Load test data ────────────────────────────────
    test_df = pd.read_csv(PROCESSED_PATH / "test.csv")
    
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    
    print(f"\nTest data loaded:")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")
    
    print(f"\nTest target distribution:")
    print(y_test.value_counts())
    
    return dt_model, rf_model, X_test, y_test


def evaluate_models(dt_model, rf_model, X_test, y_test):
    
    models = {
        'Decision Tree': dt_model,
        'Random Forest': rf_model
    }
    
    for name, model in models.items():
        
        # predictions
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # score
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n{name}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Correct predictions: {(y_pred == y_test).sum()} / {len(y_test)}")


if __name__ == "__main__":
    dt_model, rf_model, X_test, y_test = load_models_and_data()
    evaluate_models(dt_model, rf_model, X_test, y_test)

    