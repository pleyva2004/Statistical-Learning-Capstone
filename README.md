# Statistical Learning Capstone

Applied Statistics B.S. capstone senior research class at New Jersey Institute of Technology.

---

## Project 1: Insurance Charge Prediction

**Analysis**: Linear and Ridge regression to predict insurance charges
**Dataset**: Medical insurance data (1,338 observations)
**Key Finding**: Smoking status is the dominant predictor (coefficient ~23,644), followed by BMI, age, and children
**Model Performance**: R² = 0.769

📄 [Full Report](project1-insurance-analysis/insurance_analysis_report.pdf) | 📓 [Notebook](project1-insurance-analysis/ProjectOne.ipynb)

---

## Project 2: Breast Cancer Survival Analysis

**Analysis**: Cox proportional hazards regression and Kaplan-Meier estimation
**Dataset**: Haberman breast cancer survival data (306 patients, 1958-1969)
**Key Finding**: Number of positive lymph nodes is the only significant predictor (HR=1.048, p<0.001)
**Outcome**: 73.5% five-year survival rate

📄 [Full Report](project2-survival-analysis/survival_analysis_report.pdf) | 📓 [Notebook](project2-survival-analysis/ProjectTwo.ipynb)

---

## Repository Structure

```
├── data/                        # Datasets
├── project1-insurance-analysis/ # Project 1 files and plots
├── project2-survival-analysis/  # Project 2 files and plots
├── scripts/                     # Python utility scripts
└── archive/                     # Development notebooks
```

---

## Author

**Pablo Leyva**
pl33@njit.edu
B.S. Applied Statistics, New Jersey Institute of Technology
