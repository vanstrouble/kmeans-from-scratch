import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score


def normalize(data):
    """Normalize data using min-max scaling"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


def standardize(data):
    """Standardize data"""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    standar_data = (data - means) / stds
    return standar_data


def accuracy(assigned_labels, true_labels, data):
    acc = np.sum(assigned_labels == true_labels) / len(true_labels)
    # for index, element in enumerate(assigned_labels):
    #     print(f'{data[index]}, pred: {element}, expected: {true_labels[index]}')
    return acc


def graph(data, centroids=None, title="", xlabel="", ylabel="", colors=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=colors, label="Data Points")
    if centroids is not None:
        plt.scatter(
            centroids[:, 0], centroids[:, 1], c="red", marker="x", label="Centroids"
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()


def kmeans(
    data, k=3, epochs=100, labels=None, norm=True, init_method="random", umbral=0.0001
):
    """K-means algorithm"""
    dataset = normalize(data) if norm else data

    ds_size, ds_dim = dataset.shape
    centroids = generate_centroids(
        dataset, k, ds_size, labels=labels, method=init_method
    )
    # print(f"Prototypes: \n{prototypes}")

    converged = False
    epoch_count = 0
    while not converged and epoch_count < epochs:
        distances = np.linalg.norm(dataset[:, np.newaxis, :] - centroids, axis=2)
        cluster_ids = np.argmin(distances, axis=1)

        # new_centroids = np.array([np.mean(dataset[cluster_ids == j], axis=0) for j in range(k)])
        new_centroids = generate_centroids(
            dataset, k, ds_size, labels=cluster_ids, method="mean"
        )

        # graph(dataset, centroids, 'Scatter Plot with Prototypes', 'X Label', 'Y Label')

        max_centroid_diif = np.max(np.abs(centroids - new_centroids))
        if max_centroid_diif < umbral:
            converged = True
        else:
            centroids = new_centroids.copy()
            epoch_count += 1

    print(f"Epochs: {epoch_count}")
    cluster_counts = np.bincount(cluster_ids)
    for cluster_id, count in enumerate(cluster_counts):
        print(f"Cluster {cluster_id}: {count} data points")

    if labels is not None:
        acc = accuracy(cluster_ids, labels, dataset)
        print(f"Accuracy: {acc:.2f}")

    colors = [plt.cm.jet(float(i) / max(cluster_ids)) for i in cluster_ids]
    graph(
        dataset,
        centroids,
        "Scatter Plot with Clusters",
        "X Label",
        "Y Label",
        colors=colors,
    )


def compute_distance(data, k):
    distances = np.zeros((data.shape[0], k.shape[0]))
    for i, k_values in enumerate(k):
        distances[:, i] = np.linalg.norm(data - k_values, axis=1)
    return distances


def generate_centroids(data, k):
    k_indices = np.random.choice(len(data), 1, replace=False)
    centroids = data[k_indices]

    for _ in range(1, k):
        distances = np.min(compute_distance(data, centroids), axis=1)
        probs = distances / np.sum(distances)
        cumulative_probs = np.cumsum(probs)
        r = np.random.rand()
        i = np.searchsorted(cumulative_probs, r)
        centroids = np.vstack([centroids, data[i]])

    return centroids


def update_centroids(data, k, labels):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
    return new_centroids


class Kmeans:
    def __init__(self, k=3, max_ters=300) -> None:
        self.k = k

    def predict(self, data):
        self.centroids = generate_centroids(data, self.k)

        for _ in range(self.max_iters):
            self.labels = np.argmin(compute_distance(data, self.centroids), axis=1)
            new_centroids = update_centroids(data, self.k, labels=self.labels)
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        return self.labels, self.centroids


if __name__ == "__main__":
    print("Welcome to Kmeans")
    # setup = int(input("Select a dataset [1. Iris, 2. Randomized]: "))

    # if setup == 1:
    #     # Loading iris dataset
    #     iris = load_iris()
    #     iris_data = iris.data
    #     iris_labels = iris.target
    #     # kmeans(
    #     #     iris_data,
    #     #     k=3,
    #     #     epochs=100,
    #     #     norm=True,
    #     #     labels=iris_labels,
    #     #     init_method="random",
    #     # )

    # else:
    #     random_points = np.random.randint(0, 100, (100, 2))
    #     # kmeans(
    #     #     random_points,
    #     #     k=3,
    #     #     epochs=200,
    #     # )

    iris = load_iris()
    X = iris.data
    y = iris.target

    k = 3

    centroids = generate_centroids(X, k)
    print(centroids)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c="blue", label="Datos")
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        marker="X",
        s=200,
        label="Centroides iniciales",
    )
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Centroides Iniciales con KMeans++")
    plt.legend()
    plt.show()
