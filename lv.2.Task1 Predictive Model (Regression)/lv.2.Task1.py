import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
#1. Load the dataset
df = pd.read_csv('insurance.csv')
print(df.head())
print(df.isnull().sum())
print(df.info())
print("\n--- Summary Statistics ---\n")
print(df.describe())
#2. Visualize the distribution of charges
plt.figure(figsize=(6, 4))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title("Medical Charges: Smoker vs Non-Smoker")
plt.show()
#3. data preprocessing
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)
print("\n--- Data Preprocessing ---\n")
X = df.drop(columns=['charges'])
y = df['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
#4. Train a linear regression model
lr_model = LinearRegression()
print("\n--- Model Training ---\n")
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f"Mean Squared Error (MSE): {mse_lr:.2f}")
print(f"R-squared (R2 score): {r2_lr:.4f}")
features = X.columns
smoker_idx = features.get_loc('smoker')
print(f"Smoker Feature Coefficient: {lr_model.coef_[smoker_idx]:.2f}")
#5. Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
#Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
#6. Evaluate all models
print("\n--- Evaluate All Models ---\n")
def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
print("\n--- Model Comparison ---\n")
evaluate("Linear Regression", y_test, y_pred_lr)
evaluate("Decision Tree", y_test, y_pred_dt)
evaluate("Random Forest", y_test, y_pred_rf)
#7. Visual comparison
print("\n--- Visual Comparison ---\n")
model_names = ['Linear Regression', 'Decision tree', 'Random Forest']
r2_scores_list = [
    r2_score(y_test, y_pred_lr),
    r2_score(y_test, y_pred_dt),
    r2_score(y_test, y_pred_rf)
]
plt.figure(figsize=(6, 4))  # <-- Step 1: Create the window first
sns.barplot(x=model_names, y=r2_scores_list)  # <-- Step 2: Draw the plot on it
plt.title("R² Score Comparison Across Models")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.show()