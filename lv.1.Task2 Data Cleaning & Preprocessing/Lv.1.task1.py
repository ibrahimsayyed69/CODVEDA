import pandas as pd
import numpy as np
df = pd.read_csv('churn-bigml-20.csv')
df.head()
#1. Explore the dataset
print(df.info())
print(df.describe())
print(df.isnull().sum()) #all zeros, so there is no imputation needed
#2. Removing outliers
numerical_columns = [
    'Account length', 'Total day minutes', 'Total day calls', 'Total day charge',
    'Total eve minutes', 'Total eve calls', 'Total eve charge',
    'Total night minutes', 'Total night calls', 'Total night charge',
    'Total intl minutes', 'Total intl calls', 'Total intl charge',
    'Number vmail messages', 'Customer service calls'
]
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
print(f"'shape after removing outliers: {df.shape}")
#3. Encoding categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['International plan'] = le.fit_transform(df['International plan'])
df['Voice mail plan'] = le.fit_transform(df['Voice mail plan'])
df['Churn'] = le.fit_transform(df['Churn'])
df = pd.get_dummies(df, columns=['State'], drop_first=True)
#4. Normalizing numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
#5. Final dataset shape
print(df.info())
print(df.isnull().sum())
df.to_csv('churn_cleaned.csv', index=False)