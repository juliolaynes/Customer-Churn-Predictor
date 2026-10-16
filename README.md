<img width="1377" height="768" alt="image_4ec8df6" src="https://github.com/user-attachments/assets/35b29750-9fc9-4925-859b-e817192fad99" />
# Telecom Customer Churn Predictor: Transforming Structural Data into Retention Strategies

An end-to-end data science and machine learning project predicting customer cancellation (churn) using a dataset of **7,000+ customer profiles** from IBM Telecom.

---

## 🎯 Business Problem & Objective
In subscription-based models (SaaS, Telecom), customer acquisition costs (CAC) are significantly higher than retention costs. 
* **The Situation:** Identifying unsatisfied or at-risk customers manually is impossible at scale.
* **The Goal:** Ingest and clean customer usage profile data using SQL, and deploy a high-performance tree-based machine learning classifier capable of flagging at-risk accounts with high sensitivity (Recall) to trigger automated marketing retention campaigns.

---

## 🛠️ Tech Stack & Architecture
* **Data Extraction & Manipulation:** SQL (via DuckDB)
* **Data Pipelines & Wrangling:** Python, Pandas, NumPy
* **Data Visualization:** Seaborn, Matplotlib
* **Machine Learning & Valuation:** Scikit-Learn (Random Forest Classifier)

---

## 📂 Project Structure & Pipeline Steps

### 🔹 Phase 1: Advanced SQL Extraction (DuckDB)
Data was extracted from the raw layout using optimized ANSI SQL clauses. The query utilized structural conversion algorithms via `CASE WHEN` and conditional string trimming (`TRIM`) to transform structural text records (`TotalCharges`, `Churn`) into appropriate formats required for programming workflows.

### 🔹 Phase 2: Missing Data Treatment & Encoding
* **Anomaly Resolution:** Missing values resulting from zero-tenure new users were imputed with a $0 baseline balance in Python.
* **Feature Processing:** Applied **One-Hot Encoding** (`pd.get_dummies`) to map multiple categorical variables (contract terms, payment workflows, and service types) into valid sparse matrix systems.

---

## 📊 Key Analytical Insight (EDA)

Exploratory Analysis validated that contract lifecycles completely dictate churn behavior. **Month-to-month contracts experience a staggering 42.71% churn rate**, while long-term 2-year agreements drop friction down to a minimal **2.83%**.

---

## 🤖 Machine Learning Implementation & Evaluation

An optimized **Random Forest Classifier** was deployed with 100 decision estimators. To solve severe class imbalance (minority churn layer), the architecture incorporated structural class weight balancing.

### Final Classification Metrics:

* **ROC-AUC Score:** **0.8440** (Highly reliable separation)
* **Target Class (Churn) Recall:** **79%** 
* **Overall Accuracy:** **75%**

**Business Trade-off:** The model was tuned to maximize **Recall (79%)**, guaranteeing that 8 out of 10 canceling customers are caught proactively. While the precision sits at 52%, the operational cost of launching a retention offer (like a discount) is dramatically cheaper than losing an active account completely.

---

## 📈 Top Business Drivers (Feature Importance)

The non-linear ensemble algorithm ranked the top factors steering customer churn:
1. `tenure` (Customer relationship longevity) — *17.41% impact*
2. `TotalCharges` (Overall financial investment) — *12.91% impact*
3. `Contract_Two year` (Long-term contractual obligation) — *11.56% impact*
4. `MonthlyCharges` (Monthly subscription cost weight) — *8.52% impact*

---

## 🚀 How to Reproduce
1. Clone this repository.
2. Ensure the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file is in the project directory.
3. Install dependencies: `pip install pandas numpy duckdb scikit-learn seaborn matplotlib`
4. Run the notebook pipeline..
