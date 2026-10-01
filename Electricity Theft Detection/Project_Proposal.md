# Project Proposal

## Title
**Electricity Theft Detection in Distribution Networks using Machine Learning**

## Department
B.Tech — Electrical and Electronics Engineering (EEE)

## Problem Statement
Electric power distribution utilities lose a significant share of generated/purchased
energy to **Non-Technical Losses (NTL)** — primarily electricity theft through meter
tampering, unauthorized bypassing ("hooking"), and meter reversal/freezing. Industry
estimates place NTL at 15–30% of distributed energy in some regions, costing utilities
tens of billions of dollars annually. Traditional detection relies on manual, largely
random field inspections, which are slow, costly, and inefficient — most inspections
find nothing wrong, while genuine theft cases can go undetected for months. With the
rollout of smart meters (AMI), utilities now collect fine-grained consumption data that
could reveal tampering behaviour — but it is too voluminous to review manually.

## Objective
Develop an AI/ML-based solution that analyses daily smart-meter consumption data,
engineers features capturing known theft/tampering behavioural signatures, and
classifies each consumer as **Normal** or **Theft-risk**, so utilities can prioritise
field inspections using data rather than guesswork.

## Proposed Solution
1. Generate/collect a smart-meter consumption dataset (180 days of daily kWh readings
   per consumer), with realistic theft-pattern injection where real audited data is
   unavailable.
2. Preprocess the data (handle missing readings via interpolation).
3. Perform Exploratory Data Analysis to understand consumption behaviour and theft
   signatures.
4. Engineer 13 statistical/behavioural features per consumer (volatility, zero-day
   fraction, trend ratio, weekday/weekend ratio, sudden-drop count, etc.).
5. Train and compare four ML classifiers: Logistic Regression, Random Forest, Gradient
   Boosting, and SVM — using class-balancing to handle the realistic ~86:14 class
   imbalance.
6. Evaluate using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
7. Deploy the best model (Gradient Boosting) in a Streamlit web application supporting
   single-consumer and batch-CSV predictions.

## Technologies Used
- Python, Pandas, NumPy
- Matplotlib, Seaborn (visualization)
- Scikit-learn (ML models, preprocessing, evaluation)
- Jupyter Notebook
- Streamlit + Plotly (deployment/demo)
- Joblib (model persistence)

## Project Workflow
Problem Definition → Data Collection → Data Preprocessing → EDA → Feature Engineering
→ Model Training → Evaluation → Deployment/Demo

## Expected Outcome
A working prototype that flags high theft-risk consumers from their smart-meter
consumption history, achieving strong classification performance (targeted: >90%
accuracy, high F1-score on the minority theft class) and demonstrated through an
interactive web application.

## Deliverables
- [x] Project proposal (this document)
- [x] Dataset (raw smart-meter data + engineered feature dataset)
- [x] Jupyter Notebook / source code (full pipeline, executed end-to-end)
- [x] Trained ML model (.pkl) with scaler and feature list
- [x] Working prototype/demo (Streamlit web application)
- [x] Project report (Word document)
- [x] PPT presentation (12 slides)
- [ ] Final project demonstration (live viva — to be presented by the student)

## Timeline (suggested, 6–8 weeks)
| Week | Activity |
|---|---|
| 1 | Problem definition, literature review, dataset planning |
| 2 | Data collection/generation, preprocessing |
| 3 | Exploratory Data Analysis |
| 4 | Feature engineering |
| 5 | Model training and comparison |
| 6 | Evaluation, hyperparameter tuning |
| 7 | Streamlit app development |
| 8 | Report writing, presentation preparation, final demo |
