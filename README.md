
# Datathon 2025

This repository contains my solution for **Datathon 2025**.  
The project focuses on predicting **session value** based on e-commerce user behavior data.

##  Files
- `main.py` → Performs **feature engineering** on raw session data and generates the processed dataset.
- `modeling.py` → Trains machine learning models (Random Forest, XGBoost) on the engineered features and evaluates performance.
- `train.csv` → Original training dataset.
- `test.csv` → Test dataset provided by the competition.
- `submission.csv` → Final predictions formatted for submission.

##  Workflow
1. Preprocessing and feature engineering (`main.py`)
2. Model training and evaluation (`modeling.py`)
3. Prediction on test data
4. Submission file generation (`submission.csv`)

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib (for visualization)

##  Results
The final model achieved a public leaderboard score of **1442.60**.

---
