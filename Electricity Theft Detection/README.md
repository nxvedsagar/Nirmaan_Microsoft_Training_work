# Electricity Theft Detection in Distribution Networks using Machine Learning
### B.Tech EEE Capstone Project

## 📁 Project Structure
```
theft_detection/
├── Project_Proposal.md                        # Project proposal document
├── data/
│   ├── generate_dataset.py                     # Generates raw smart-meter dataset
│   ├── feature_engineering.py                  # Converts raw data → ML features
│   ├── smart_meter_data.csv                    # Raw dataset (3000 consumers x 180 days)
│   └── theft_features_dataset.csv              # Engineered feature dataset (ML-ready)
├── notebook/
│   ├── theft_detection_pipeline.py             # Source (jupytext percent format)
│   └── theft_detection_pipeline.ipynb          # Full executed Jupyter notebook
├── model/
│   ├── theft_detection_model.pkl               # Trained Gradient Boosting classifier
│   ├── scaler.pkl                              # StandardScaler (for LR/SVM if used)
│   └── feature_columns.json                    # Ordered list of feature names
├── app/
│   └── streamlit_app.py                        # Streamlit demo application
└── report_assets/
    ├── *.png                                   # All charts/figures from the notebook
    ├── model_comparison.csv                    # Model comparison results table
    ├── Theft_Detection_Project_Report.docx     # Full project report (Word)
    └── Theft_Detection_Presentation.pptx       # Presentation slides (PowerPoint)
```

## 🚀 How to Run

### 1. Re-run the full pipeline (optional — outputs already generated)
```bash
cd data
python3 generate_dataset.py        # regenerate raw dataset
python3 feature_engineering.py     # regenerate feature dataset

cd ../notebook
jupyter nbconvert --to notebook --execute --inplace theft_detection_pipeline.ipynb
```

### 2. Launch the Streamlit demo app
```bash
cd app
pip install streamlit plotly joblib scikit-learn pandas numpy
streamlit run streamlit_app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`).

**Try it with:**
- **Single Consumer Check** tab → pick a simulated pattern (e.g. "Sudden Drop") and click
  "Generate & Predict" to see the model flag it as theft-risk.
- **Batch Upload** tab → upload `data/smart_meter_data.csv` directly to see predictions
  across all 3,000 consumers.

## 📊 Model Performance Summary
| Model | Accuracy | F1-Score (Theft) | ROC-AUC |
|---|---|---|---|
| **Gradient Boosting (deployed)** | **97.17%** | **0.894** | **0.991** |
| Random Forest | 97.00% | 0.886 | 0.990 |
| SVM (RBF) | 93.50% | 0.800 | 0.984 |
| Logistic Regression | 91.33% | 0.732 | 0.966 |

## 📝 Notes for Real-World Deployment
This project uses a **synthetically generated but realistically structured** dataset
(modelled on public AMI dataset schemas such as SGCC), since real utility theft data is
access-restricted. To deploy in production:
1. Replace `data/smart_meter_data.csv` with actual AMI billing records merged with
   confirmed inspection/audit outcomes (ground-truth theft labels).
2. Re-run `feature_engineering.py` and the training notebook on the real data.
3. Validate feature distributions still separate the classes as expected (see EDA
   section of the notebook) before trusting predictions operationally.
