import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer


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


def compute_inertia(data, centroids, labels):
    inertia = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]

        if len(cluster_points) > 0:
            distances = compute_distance(cluster_points, np.array([centroid]))[:, 0]
            inertia += np.sum(distances**2)

    return inertia


class Kmeans:
    def __init__(self, k=3, max_iters=300) -> None:
        self.k = k
        self.max_iters = max_iters
        self.inertia = None
        self.iter_num = 0

    def predict(self, data, inertia=False):
        self.centroids = generate_centroids(data, self.k)

        for _ in range(self.max_iters):
            self.labels = np.argmin(compute_distance(data, self.centroids), axis=1)
            new_centroids = update_centroids(data, self.k, labels=self.labels)
            new_inertia = compute_inertia(data, new_centroids, self.labels)

            if self.inertia is not None and np.allclose(self.inertia, new_inertia):
                break
            self.centroids = new_centroids
            self.inertia = new_inertia

            self.iter_num += 1

        if inertia:
            return self.labels, self.centroids, self.inertia

        return self.labels, self.centroids

    def plot(self, data):
        plt.scatter(data[:, 0], data[:, 1], c=self.labels, label="Data Points")
        plt.scatter(
            self.centroids[:, 0],
            self.centroids[:, 1],
            c="red",
            marker="x",
            label=f"Centroids | Iter: {self.iter_num}",
        )
        plt.title("Scatter Plot with Clusters")
        plt.xlabel("X Label")
        plt.ylabel("Y Label")
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_normalized = Normalizer().fit_transform(X)

    k = 3

    k_means = Kmeans(k=k)
    labels, centroids = k_means.predict(X_normalized)
    k_means.plot(X_normalized)
