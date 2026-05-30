# Key findings:
# 1. Target is imbalanced: 700 good (70%), 300 bad (30%)
# 2. checking_account is strongest single predictor
# 3. duration and credit_amount correlate with bad risk
# 4. personal_status combines gender + marital status — fairness concern
# 5. No true missing values — absences are meaningful categories

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))       #--> Add parent directory to sys.path to import config.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PATH, FIGURES_PATH, TARGET, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Data loaded successfully.")
    print(f"Dataset shape: {df.shape}")
    print(f"\n all Columns:\n {df.columns.tolist()}")
    print(f"\n first 5 rows: \n {df.head()}")
    return df

def inspect_data(df):
    # data typrs
    print("Data Types: ", df.dtypes)

    # missing values 
    print("\n Missing values")
    print(df.isnull().sum())

    # class distribution 
    print("\nClass Distibution")
    cnt = df[TARGET].value_counts()
    print(cnt)

"""
# save class distribution graph 
plt.figure()
ax = sns.countplot(
    x= TARGET,
    data= load_data(),
    hue= TARGET,
    palette= {0 : "#2ecc71", 1: "#e74c3c"},
    legend=False
)
plt.title("Class Distribution — Good Risk vs Bad Risk")
plt.xlabel("Credit Risk  (0 = Good, 1 = Bad)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES_PATH / "class_distribution.png", dpi=150)
plt.close()
print("\nChart saved: class_distribution.png")

"""

def analyze_numerical(df):
    # summery 
    print("\n numerical feature summery")
    print(df[NUMERICAL_FEATURES].describe().round(2))

    # mean by class
    print("\n Mean by class")
    print(df.groupby(TARGET)[NUMERICAL_FEATURES].mean().round(2).T)

    print("\n std deviation by class")
    print(df.groupby(TARGET)[NUMERICAL_FEATURES].std().round(2).T)

def analyze_categorical(df):

    for feature in CATEGORICAL_FEATURES:
        print(f"\n {'='*30}")
        print(f"Feature: {feature}")
        print(f"\n {'='*30}")

        print(f"unique values: {df[feature].unique()}")

        print("Value counts: ", df[feature].value_counts())

        print(f"\nDistribution by class (%):")
        cross = pd.crosstab(
            df[feature],
            df[TARGET],
            normalize='index'
        ).round(3) * 100
        cross.columns = ['Bad Risk %','Good Risk %']
        print(cross)



if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    analyze_numerical(df)
    analyze_categorical(df)
  



