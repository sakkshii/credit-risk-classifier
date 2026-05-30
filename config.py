# Every project has values that get used in multiple places — file paths, model hyperparameters, the random seed, the test split size. If you hardcode these directly inside eda.py, train.py, preprocess.py, you end up with the same number scattered across five files

# To avoid this, we can create a config.py file that contains all of these values in one place. Then, if we need to change something, we only have to change it in one place.

from pathlib import Path

# Define the base directory for the project
BASE_DIR = Path(__file__).parent
# Define paths for data and figures
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'GermanCredit.csv'
FIGURES_PATH = BASE_DIR / 'reports' / 'figures'
PROCESSED_PATH = BASE_DIR / 'data' / 'processed' 
MODELS_PATH = BASE_DIR / 'models'

# TARGET 
TARGET = 'credit_risk'

# Target encoding
GOOD_RISK = 1
BAD_RISK  = 0

# feature columns
NUMERICAL_FEATURES = ['duration', 'amount', 'installment_rate', 'present_residence', 'age', 'number_credits', 'people_liable']

CATEGORICAL_FEATURES = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker"
]

#model hyperparameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_DEPTH = 5
N_ESTIMATORS = 100

# cost matrix 
COST_FN = 5  # Cost of false negative (predicting good when it's actually bad)
COST_FP = 1  # Cost of false positive (predicting bad when it's actually good)

ORDINAL_FEATURES = [
    'status',
    'savings',
    'employment_duration',
    'credit_history'
]

ORDINAL_ORDERS = [
    # status
    ['no checking account',
        '... < 100 DM',
        '0 <= ... < 200 DM',
        '... >= 200 DM / salary assignments for at least 1 year'], 
    # saving
    ['unknown/no savings account',
        '... < 100 DM',
        '100 <= ... < 500 DM',
        '500 <= ... < 1000 DM',
        '... >= 1000 DM'], 
    # employment durstion
    ['unemployed',
        '... < 1 year',
        '1 <= ... < 4 years',
        '4 <= ... < 7 years',
        '... >= 7 years'],
    # credit history
    [ 'no credits taken/all credits paid back duly',
        'all credits at this bank paid back duly',
        'existing credits paid back duly till now',
        'delay in paying off in the past',
        'critical account/other credits existing']
]


NOMINAL_FEATURES = [
    'purpose',
    'personal_status_sex',
    'other_debtors',
    'property',
    'other_installment_plans',
    'housing',
    'job',
    'telephone',
    'foreign_worker'
]

#  Hyperparameter grids 
DT_PARAM_GRID = {
    'max_depth':        [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'criterion':        ['gini', 'entropy']
}

RF_PARAM_GRID = {
    'n_estimators':     [100, 200, 300],
    'max_depth':        [3, 4, 5, 6, 7],
    'min_samples_split': [2, 5, 10],
    'max_features':     ['sqrt', 'log2']
}

CV_FOLDS     = 5
SCORING      = 'roc_auc'

# ── Best parameters from tuning ───────────────────────
DT_BEST_PARAMS = {
    'criterion':         'entropy',
    'max_depth':         4,
    'min_samples_leaf':  1,
    'min_samples_split': 2
}

RF_BEST_PARAMS = {
    'max_depth':         6,
    'max_features':      'log2',
    'min_samples_split': 2,
    'n_estimators':      200
}

# ── Evaluation ────────────────────────────────────────
DEFAULT_THRESHOLD = 0.5