import matplotlib.pyplot as plt

def plot_customer_clusters(df):

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score")
    plt.title("Customer Segmentation")

    plt.colorbar(scatter)

    plt.savefig(
        "outputs/customer_clusters.png"
    )

    plt.close()


def plot_pca_clusters(
        pca_data,
        labels
):

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=labels
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Customer Clusters")

    plt.colorbar(scatter)

    plt.savefig(
        "outputs/pca_clusters.png"
    )

    plt.close()