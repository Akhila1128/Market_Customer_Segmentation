from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(data):

    inertia = []

    for k in range(1, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(data)

        inertia.append(model.inertia_)

    plt.figure(figsize=(8,5))

    plt.plot(
        range(1,11),
        inertia,
        marker='o'
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")

    plt.savefig("outputs/elbow_method.png")

    plt.close()


def apply_kmeans(data):

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(data)

    return labels