# Case Study No. 124: Content Popularity Analysis

> **Supervised Machine Learning System for Digital Content Engagement Forecasting**

This project provides a clean, fully functional machine learning system that analyzes online article features to estimate whether content is likely to attract high reader engagement ($\ge 1,400$ shares).

---

## 📁 Project Structure

```text
AIML Exam 2/
├── app.py                     # Streamlit Web Application (Interactive Dashboard & Predictor)
├── train_and_evaluate.py      # End-to-end ML Pipeline (Training, Evaluation, Model Saving)
├── generate_notebook.py       # Helper script to generate clean notebook.ipynb
├── notebook.ipynb             # Jupyter Notebook with step-by-step EDA & Model Training
├── PROJECT_REPORT.md          # Comprehensive Academic & Technical Project Report
├── PRESENTATION_SLIDES.md     # 10-Slide Presentation Deck
├── data/
│   ├── OnlineNewsPopularity.csv   # UCI Dataset (39,644 articles)
│   ├── OnlineNewsPopularity.names # Dataset schema & documentation
│   └── sample_test.csv            # Sample test rows for quick testing
└── models/
    ├── best_model.joblib          # Trained Random Forest Classifier
    ├── all_models.joblib          # All trained models (LogReg, DecisionTree, RF)
    ├── scaler.joblib              # Fitted StandardScaler
    ├── feature_names.json         # Exact feature column order
    ├── model_comparison.json      # Precomputed benchmark metrics & confusion matrices
    └── eda_summary.json           # Precomputed EDA aggregates for instant dashboard load
```

---

## 🚀 How to Run the Project

### 1. (Optional) Re-train the Models
The models and precomputed evaluations are already saved in the `models/` directory. If you want to re-run the full training pipeline:

```bash
python3 train_and_evaluate.py
```

### 2. Launch the Streamlit Application
To start the interactive web app:

```bash
python3 -m streamlit run app.py
```
*(or `streamlit run app.py` if streamlit is in your shell PATH)*

Once running, the application will open automatically in your browser at `http://localhost:8501`.

---

## 🎯 Deliverables Summary

| Deliverable | Location in Project | Key Details |
| :--- | :--- | :--- |
| **Problem Definition** | `PROJECT_REPORT.md` (Sec. 1), `app.py` (Tab 1) | Binary classification at 1,400 median shares |
| **Dataset & Documentation**| `data/`, `PROJECT_REPORT.md` (Sec. 2) | UCI dataset, 39,644 records, 17 curated features |
| **EDA & Visualizations** | `app.py` (Tab 2), `notebook.ipynb` | Channel virality, weekend effect, media impact |
| **Preprocessing** | `train_and_evaluate.py`, `PROJECT_REPORT.md` | Cleaning, 80/20 stratified split, StandardScaler |
| **Model Development** | `train_and_evaluate.py`, `models/` | Logistic Regression, Decision Tree, Random Forest |
| **Evaluation & Results** | `PROJECT_REPORT.md` (Sec. 5 & 6), `app.py` (Tab 3)| Accuracy, Precision, Recall, F1, ROC-AUC, CM |
| **Streamlit Application**| `app.py` | Interactive dashboard & live article predictor |
| **Report & Presentation**| `PROJECT_REPORT.md`, `PRESENTATION_SLIDES.md` | Academic report & 10-slide presentation deck |

---

## 📊 Summary of Model Performance

Benchmark results on **7,929 test samples**:

* **Random Forest (Best Model)**:
  * Accuracy: **65.22%**
  * Recall: **72.54%**
  * ROC-AUC: **0.7088**
* **Logistic Regression**: Accuracy: **64.71%** | ROC-AUC: **0.6883**
* **Decision Tree**: Accuracy: **64.23%** | ROC-AUC: **0.6796**
