# 🏘️ Real Estate Market Segmentation using Unsupervised K-Means & PCA
## Task 3: Clustering (Unsupervised Learning)

An end-to-end Machine Learning pipeline that groups complex housing and neighborhood data into distinct, interpretable property classes. Since the housing dataset features multi-dimensional attributes (such as crime rates, building age, average rooms, tax rates, etc.), this pipeline leverages **Principal Component Analysis (PCA)** to reduce the 14-dimensional feature space down to 2D for intuitive, high-impact visualization.

---

## 📌 Project Objectives & Deliverables
1. **Apply K-Means Clustering** to segment housing market tracts based on environmental, physical, and socio-economic features.
2. **Determine Optimal Clusters ($k$)** mathematically using the **Elbow Method** (Within-Cluster Sum of Squares - WCSS).
3. **Dimensionality Reduction** utilizing **PCA** (Principal Component Analysis) to visualize complex multidimensional data into an easily interpretable 2D plane.
4. **Interpret Cluster Characteristics** to generate actionable business insights for real estate sectors.

---

## 🛠️ Tech Stack & Key Libraries
- **Language:** Python 3.x
- **Data Wrangling:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning Engine:** `scikit-learn`
  - Preprocessing: `StandardScaler`
  - Clustering Model: `KMeans`
  - Dimensionality Reduction: `PCA`

---

## 📂 Dataset Architecture
The pipeline operates on the classic **Boston Housing Dataset** (`4) house Prediction Data Set.csv`), consisting of **506 unique geographical tracts** mapping 14 continuous attributes:
* **`CRIM`**: Per capita crime rate by town
* **`ZN`**: Proportion of residential land zoned for large lots
* **`INDUS`**: Proportion of non-retail business acres per town
* **`CHAS`**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
* **`NOX`**: Nitric oxides concentration
* **`RM`**: Average number of rooms per dwelling
* **`AGE`**: Proportion of owner-occupied units built prior to 1940
* **`DIS`**: Weighted distances to five employment centers
* **`RAD`**: Index of accessibility to radial highways
* **`TAX`**: Full-value property-tax rate per $10,000
* **`PTRATIO`**: Pupil-teacher ratio by town
* **`B`**: Proportion of residents of specific demographics
* **`LSTAT`**: Percentage of lower status of the population
* **`MEDV`**: Median value of owner-occupied homes (Target evaluation in $1000s)

---