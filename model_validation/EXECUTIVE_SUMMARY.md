# Credit Risk Model Validation — Executive Summary

**Project goal:**  
Develop and validate a baseline credit-risk probability of default (PD) model in Python, following bank-grade validation practices.

---

## Key Outcomes

- **Champion model:** Logistic Regression with log-transformed income & loan amount  
- **Performance (sealed TEST set):**  
  - ROC AUC = **0.675**  
  - PR AUC = **0.142**  
  - KS = **0.256**  
  - Brier = **0.229**  

- **Business KPIs:**  
  - Top 10% highest-risk segment captures **24.5%** of defaults  
  - Lift @ top decile = **2.45×**  

- **Interpretability:**  
  - Larger loans ↑ risk (OR ≈ 1.7)  
  - Higher income & credit score ↓ risk  
  - Unemployment & certain regions ↑ risk  

- **Calibration:** Raw logit underpredicted probabilities; Platt/Isotonic calibration corrected this.  
- **Deliverables:** Packaged as reproducible Jupyter notebooks, Python modules, and zipped artifact bundle.

---

## Repo Highlights
- **End-to-end notebook**: [01_end_to_end_model_validation.ipynb](notebooks/01_end_to_end_model_validation.ipynb)  
- **Reusable package**: `src/model_validation/` (diagnostics, training, validation, monitoring)  
- **Final deliverables**: Metrics, gains curves, coefficients, and HTML report in `reports/artifacts/final/`

---
