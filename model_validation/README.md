# Credit Risk Model Validation (Logistic Regression)

This repository contains a complete workflow for developing, validating, and packaging a logistic regression credit risk model.  
It is designed as a learning + showcase project for Swedbank model validation and general credit risk analytics.

---

## 📂 Project Structure

data/               # Input datasets (synthetic sample_data.csv)  
notebooks/          # Jupyter walkthroughs (end-to-end + experiments)  
reports/            # Outputs: metrics, plots, packaged deliverables  
src/                # Python package code (diagnostics, training, validation)  
requirements_*.txt  # Environment setup  
pyproject.toml      # Optional: modern packaging metadata  
CHANGELOG.md        # Development notes  

---

## 🔑 Workflow Steps

1. **Diagnostics**  
   - Missingness, distributions, correlations  
   - Hosmer–Lemeshow, VIF, influence analysis  

2. **Training & Validation**  
   - Split data 70/15/15  
   - Compare baseline vs log-transformed features  
   - Select champion using ROC AUC, PR AUC, KS, Brier  

3. **Test Evaluation (sealed)**  
   - Evaluate champion on unseen TEST split  
   - Deciles & Gains curve (business KPIs)  

4. **Threshold Selection**  
   - Precision, Recall, F1 vs threshold  
   - Best cutoff saved in JSON/HTML report  

5. **Final Packaging**  
   - JSON summary, HTML report, all artifacts  
   - Deliverable .zip for portability  

---

## 📊 Example Results

Champion: Log-variant Logistic Regression

- Validation ROC AUC = 0.672  
- Test ROC AUC = 0.675, PR AUC = 0.142  
- Lift @ top decile = 2.45×  
- Capture @ top 10% = 24.5% of all defaults  

---

## ⚙️ Usage

Install dependencies:

pip install -r requirements_base.txt

Run main notebook:

jupyter notebook notebooks/01_end_to_end_model_validation.ipynb

Or run directly as package scripts:

python -m model_validation.training.train_model

---

## 📦 Deliverables

The final packaged output includes:

- Trained champion model (.joblib)  
- Validation & test metrics (.csv)  
- Deciles table & Gains curve (.csv / .png)  
- Final JSON + HTML report  
- README summary  

All bundled under:  
/reports/artifacts/final/deliverable_YYYYMMDD_HHMMSS.zip

---

## 🚀 Next Steps

- Add regularization (L1/L2) for stability  
- Add monitoring (data drift, stability) under src/model_validation/monitoring/  
- Extend to IFRS 9 / IRB validation approaches  

---

Author: Renee Bahman (with assistance from ChatGPT)  
Built with Python, scikit-learn, pandas, matplotlib
"""
