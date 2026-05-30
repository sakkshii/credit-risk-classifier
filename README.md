# 💳 Credit Risk Classifier

A machine learning project that predicts whether a loan applicant
is a Good Risk or Bad Risk using the German Credit dataset.

---

## 🎯 Problem Statement

Banks need to assess whether a loan applicant will repay their loan.
Approving a bad loan is 5× more costly than rejecting a good one.
This model helps loan officers make data-driven decisions.

---

## 📊 Dataset

- **Source**: German Credit Risk dataset (UCI Machine Learning Repository)
- **Size**: 1,000 applicants, 20 features
- **Target**: Credit Risk (Good / Bad)
- **Class distribution**: 70% Good Risk, 30% Bad Risk

---

## 🏗️ Project Structure

credit-risk-classifier/
├── src/
│ ├── eda.py # Exploratory data analysis
│ ├── preprocess.py # Data cleaning and encoding
│ ├── train.py # Model training and tuning
│ ├── evaluate.py # Model evaluation
│ └── predict.py # Prediction logic
├── app.py # Streamlit web application
├── config.py # Project configuration
├── requirement.txt # Dependencies
└── README.md

---

## ⚙️ How It Works

### 1. Preprocessing

- Split data 80/20 (stratified)
- Ordinal encoding for ordered categories
- One-hot encoding for nominal categories
- SMOTE to handle 70/30 class imbalance

### 2. Models Trained

- Decision Tree (baseline)
- Random Forest (production model)

### 3. Hyperparameter Tuning

- GridSearchCV with 5-fold cross-validation
- Optimised for ROC-AUC score

---

## 📈 Results

| Model         | ROC-AUC | Correct Predictions |
| ------------- | ------- | ------------------- |
| Decision Tree | 0.76    | 132 / 200           |
| Random Forest | 0.77    | 146 / 200           |

---

## 🔍 Key Findings from EDA

- Bad Risk applicants borrow **32% more** on average (3,938 vs 2,985 DM)
- Bad Risk applicants have **longer loan durations** (24.86 vs 19.21 months)
- **Checking account status** is the strongest single predictor
- **Savings account balance** strongly correlates with creditworthiness

---

## ⚠️ Fairness Note

The dataset encodes gender and marital status as a single combined
feature. All female applicants share one category regardless of
marital status, while male applicants have three distinct categories.
This raises a fair lending concern and would require compliance
review before deploying in a real credit decision system.

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/sakkshii/credit-risk-classifier.git
cd credit-risk-classifier
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/uciml/german-credit)
and place it at `data/raw/german_credit.csv`

### 5. Run the pipeline

```bash
python src/preprocess.py
python src/train.py
python src/evaluate.py
```

### 6. Launch the app

```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool             | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Core language             |
| pandas / numpy   | Data manipulation         |
| scikit-learn     | ML models and evaluation  |
| imbalanced-learn | SMOTE for class imbalance |
| Streamlit        | Web application           |
| joblib           | Model serialisation       |

---

## 👤 Author

**Sakshi**

- GitHub: [@sakkshii](https://github.com/sakkshii)
