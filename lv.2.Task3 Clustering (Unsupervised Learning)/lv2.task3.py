import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
print("All libraries imported successfully.")
#1. Column names
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
#2. Load the dataset
df = pd.read_csv('4) house Prediction Data Set.csv', sep=r'\s+', header=None, names=column_names)
print("Dataset loaded successfully.")
print(df.head())
#3. clean the dataset
print(df.info())
print(df.isnull().sum())
print(df.describe())
print("Dataset cleaned successfully.")
#4. Clustering
X = df.copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=column_names)
print("Data scaled successfully.")
print(X_scaled_df.head())
#5. Determine the optimal number of clusters using the elbow method
k_means = range(2, 11)  # silhouette score is undefined for k=1
wcss = []
silhouette_scores = []


for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    
# Plot the elbow method graph
plt.figure(figsize=(8, 5))
plt.plot(list(range(2, 11)), wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (inertia)')
plt.xticks(list(range(2, 11)))
plt.grid(True)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
plt.show()

# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(list(range(2, 11)), silhouette_scores, marker='o', linestyle='--', color='g')
plt.title('Silhouette Score by Number of Clusters')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(list(range(2, 11)))
plt.grid(True)
plt.tight_layout()
plt.savefig('silhouette_scores.png', dpi=150)
plt.show()

best_k = list(range(2, 11))[int(np.argmax(silhouette_scores))]
print(f"best k by silhouette score: {best_k} (score = {max(silhouette_scores):.3f})")

optimal_k = best_k

kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels
print(f"KMeans fitted with k={optimal_k}. Cluster Labels assigned successfully.")
print(df[['RM', 'AGE', 'MEDV', 'Cluster']].head(10))
#7. Visualize the clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = df['Cluster']
print(f"Explained variance ratio by PCA: {pca.explained_variance_ratio_}")
#8. Plot the PCA clusters
plt.figure(figsize=(10, 7))
sns.scatterplot(
    x='PCA1', y='PCA2',
      hue='Cluster',
        palette='viridis',
          data=pca_df,
            s=70,
    alpha=0.8,
      edgecolor='k'
)
plt.title(f'House Data Clusters (PCA) - k={optimal_k}')
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Cluster/Segment')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('pca_clusters.png', dpi=150)
plt.show()
#7 interpret the clusters
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print("\nCluster profile (mean of each feature per cluster):")
print(cluster_summary)
print("\nCluster sizes:")
print(df['Cluster'].value_counts().sort_index())
for c in sorted(df['Cluster'].unique()):
    subset = df[df['Cluster'] == c]
    print(f"\nCluster {c} {len(subset)} records:")
    print(f" Avg MEDV (home value): {subset['MEDV'].mean():.2f}")
    print(f" Avg RM (rooms): {subset['RM'].mean():.2f}")
    print(f" Avg AGE (building age): {subset['AGE'].mean():.2f}")
    print(f" Avg CRIM (crime rate): {subset['CRIM'].mean():.2f}")
print("\nDone. Review 'cluster_summary' above to write a short narrative, e.g.:")
print(" - Cluster 0: larger, newer homes with higher values and lower crime -> 'premium suburbs'")
print(" - Cluster 1: smaller, older homes with lower values and higher crime -> 'urban core'")
print(" - Cluster 2: mid-range accross most features -> 'mixed/suburban middle tier'")
