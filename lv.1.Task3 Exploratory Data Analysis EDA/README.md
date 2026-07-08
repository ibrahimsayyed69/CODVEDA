# Level 1 - Task 3: Exploratory Data Analysis (EDA)

A comprehensive exploratory data analysis on a Telecom Customer Churn dataset to uncover underlying structures, anomalies, and feature correlations. This project was completed as part of the Codveda Data Science Internship.

---

## 📋 Project Summary
This project walks through the complete data science lifecycle for initial data exploration. It builds an automated pipeline that imports a dataset, cleanses/audits missing entries, calculates key descriptive statistics, visualizes distributions, identifies structural outliers, and maps feature correlations to understand what drives customer churn.

### Technical Toolkit
* **Language:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn

---

## 📽️ Execution & Output Demo
Below is a demonstration of the script executing flawlessly in the terminal environment, generating the summary statistics, variance calculations, and the final automated text report.
[Watch the Execution Demo Video](

https://github.com/user-attachments/assets/41311b1f-98ef-486b-ad05-ef0585c7cc72

)

---

## 📊 Visualizations & Insights

### 1. Customer Service Calls Distribution
Most customers make between 0 and 2 calls to customer service. However, the distribution is heavily right-skewed, trailing off to a maximum of 8 calls. This subset of frequent callers serves as a critical focus group for retention teams.
![Customer Service Calls](<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/788802ae-298b-4c14-8c63-9fdd931e6803" />
)

### 2. Outlier Detection (Total Day Minutes)
The box plot highlights normal usage variations across the 667 rows, but explicitly identifies extreme high-end and low-end statistical outliers—such as users spending over 320 minutes on daytime calls.
![Boxplot Outliers](<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/99cc3735-bcf9-4280-91aa-697e903b493a" />
)

### 3. Feature Correlation Matrix
The heatmap shows a perfect positive correlation coefficient of **1.00** between call minutes and charges across all time segments, proving a rigid linear billing structure. More importantly, it reveals that `Customer service calls` and `Total day minutes/charges` have the highest relative positive correlation with the `Churn` flag (0.23–0.24).
![Correlation Heatmap](<img width="1200" height="642" alt="image" src="https://github.com/user-attachments/assets/ff9962ba-b3e3-4b9f-8a1c-a85eb9b207de" />
)

---

## 📑 Automated EDA Report Findings

The script dynamically writes the final executive findings directly to a text file (`EDA_Report.txt`). A summary of the generated report includes:

* **Data Integrity:** 0 missing values across all columns, making the data highly reliable.
* **Customer Profile:** The average customer has an account length of ~103 days and makes 1.56 customer service calls.
* **Key Takeaway:** Heavy daytime users and customers frequently escalating issues to support are your highest-risk profiles for churning.
