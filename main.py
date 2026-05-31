import pandas as pd

from src.data_preprocessing import preprocess_data
from src.rfm_analysis import create_rfm
from src.pca_analysis import apply_pca
from src.clustering import elbow_method
from src.clustering import apply_kmeans
from src.visualization import plot_customer_clusters
from src.visualization import plot_pca_clusters


print("="*50)
print(" MARKET CUSTOMER SEGMENTATION PROJECT ")
print("="*50)

# Step 1
df = preprocess_data()

print("\nDataset Loaded Successfully")
print(df.head())

# Step 2
df = create_rfm(df)

print("\nRFM Analysis Completed")

# Step 3
features = df[
    [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)",
        "Recency",
        "Frequency",
        "Monetary"
    ]
]

# Step 4
pca_data = apply_pca(features)

print("\nPCA Applied Successfully")

# Step 5
elbow_method(pca_data)

print("\nElbow Method Graph Saved")

# Step 6
labels = apply_kmeans(pca_data)

df["Cluster"] = labels

print("\nK-Means Clustering Completed")

# Step 7
cluster_report = df.groupby("Cluster").mean(
    numeric_only=True
)

cluster_report.to_csv(
    "outputs/cluster_report.csv"
)

print("\nCluster Report Generated")

# Step 8
plot_customer_clusters(df)

plot_pca_clusters(
    pca_data,
    labels
)

print("\nGraphs Generated Successfully")

print("\nCluster Report:\n")
print(cluster_report)

print("\nProject Executed Successfully")