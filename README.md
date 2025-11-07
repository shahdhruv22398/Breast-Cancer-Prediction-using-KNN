Notebook link - https://colab.research.google.com/drive/1yegBrGI1eAAfReFFJ_4_HHq5XlL190Jo?usp=sharing

# **Executive Summary**

I built a machine learning model to predict whether a breast tumor is **benign or malignant** using the **Breast Cancer Wisconsin dataset**. 
I explored and cleaned the data, analyzed feature correlations, and applied **scaling** to standardize inputs. 
I then trained and evaluated models using **scikit-learn**, achieving **high accuracy (≈95%)**. Through cross-validation, I confirmed the model’s reliability and generalization. 
This project strengthened my understanding of end-to-end ML workflows — from **EDA and preprocessing** to **model training, evaluation, and optimization** — and demonstrated how data-driven insights can support faster, more accurate medical diagnoses.

---

# **Problem Statement**

Early and accurate breast cancer diagnosis is critical to improving patient outcomes, yet traditional detection methods can be time-consuming and dependent on subjective clinical interpretation. 
The challenge is to build a reliable, data-driven model that can quickly distinguish between **benign and malignant tumors** using measurable cell-level features, 
helping clinicians prioritize high-risk cases and support faster, evidence-based decision-making.

---

# **Methodology**

- **Data Exploration & Cleaning:** Inspected dataset structure, checked class distribution, assessed missing values, and analyzed feature correlations (Pandas, NumPy, Seaborn).
- **Feature Scaling:** Applied **StandardScaler** to normalize numeric features for accurate distance measurement in KNN (scikit-learn).
- **Model Training & Evaluation:** Trained a **K-Nearest Neighbors** classifier and experimented with different values of *k* to identify the optimal model.
- **Model Validation:** Used **train-test split and cross-validation** to ensure performance consistency and reduce overfitting risks.

---

# **Skills**

- **Python** (Pandas, NumPy)
- **Data Cleaning & EDA**
- **Data Visualization** (Matplotlib, Seaborn)
- **Feature Scaling** (StandardScaler)
- **Classification Modeling** (KNN)
- **Model Evaluation** (Accuracy, Confusion Matrix)
- **Cross-Validation & Train-Test Splitting**
- **Machine Learning Workflow & Interpretation**

---

# **Results**
- Achieved **~95% accuracy**, demonstrating strong predictive performance.
- Model correctly distinguished between **benign and malignant** tumors with minimal misclassification.
- Feature scaling had a **significant positive impact** on model stability and accuracy.

# **Recommendations**
- Incorporate **additional models** (e.g., Logistic Regression, Random Forest) for comparative benchmarking.
- Use **GridSearchCV** in future iterations to automate optimal **k** selection.

---

# **Next Steps**
- Expand the analysis by comparing KNN with other classification models (e.g., Logistic Regression, SVM, Random Forest).
- Perform deeper **hyperparameter tuning** (e.g., optimal k-value search) to further improve accuracy.
- Build a **Streamlit** or **Flask** app to make the model interactive and user-friendly.
- Explore feature importance / dimensionality reduction (PCA) to simplify the model while maintaining performance.

Notebook link - https://colab.research.google.com/drive/1yegBrGI1eAAfReFFJ_4_HHq5XlL190Jo?usp=sharing
