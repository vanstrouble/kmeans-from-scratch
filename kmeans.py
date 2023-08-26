import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score


def normalize(data):
    """Normalize data"""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    normalized_data = (data - means) / stds
    return normalized_data

def generate_centroids(data, k, ds_size, labels=None, use_labels=False):
    """Initializes prototypes"""
    if use_labels is False:
        # Random choice
        p_indices = np.random.choice(ds_size, k, replace=False)
        return data[p_indices]
    else:
        # Mean choice
        centroids = []
        for label in np.unique(labels):
            label_data = data[labels == label]
            class_mean = np.mean(label_data, axis=0)
            centroids.append(class_mean)

        return np.array(centroids)

def kmeans_plus_plus_init(data, k):
    centroids = [data[np.random.choice(len(data))]]  # Initialize with random centroids

    for _ in range(1, k):
        distances = np.array([np.min(np.linalg.norm(data - centroid, axis=1) ** 2) for centroid in centroids])
        sum_distances = np.sum(distances)
        probabilities = distances / sum_distances if sum_distances > 0 else np.ones(len(data)) / len(data)
        next_centroid_idx = np.random.choice(len(data), p=probabilities)
        centroids.append(data[next_centroid_idx])

    return np.array(centroids)

def accuracy(assigned_labels, true_labels):
    ari = adjusted_rand_score(true_labels, assigned_labels)
    return ari

def graph(data, centroids=None, title='', xlabel='', ylabel='', colors=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=colors, label='Data Points')
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def kmeans(data, k=3, epochs=100, labels=None, norm=True, use_labels=False):
    """K-means algorithm"""
    dataset = normalize(data) if norm else data

    ds_size, ds_dim = dataset.shape
    centroids = kmeans_plus_plus_init(dataset, k) if not use_labels else generate_centroids(dataset, k, ds_size, labels=labels, use_labels=True)
    # print(f"Prototypes: \n{prototypes}")

    converged = False
    epoch_count = 0
    while not converged and epoch_count < epochs:
        distances = np.linalg.norm(dataset[:, np.newaxis, :] - centroids, axis=2)
        cluster_ids = np.argmin(distances, axis=1)

        new_centroids = np.array([np.mean(dataset[cluster_ids == j], axis=0) for j in range(k)])

        # graph(dataset, centroids, 'Scatter Plot with Prototypes', 'X Label', 'Y Label')

        if np.allclose(centroids, new_centroids):
            converged = True
        else:
            centroids = new_centroids.copy()
            epoch_count += 1

    print(f"Epochs: {epoch_count}")
    cluster_counts = np.bincount(cluster_ids)
    for cluster_id, count in enumerate(cluster_counts):
        print(f"Cluster {cluster_id}: {count} data points")

    a = accuracy(cluster_ids, labels)
    print(f'Accuracy: {a:.2f}')

    colors = [plt.cm.jet(float(i) / max(cluster_ids)) for i in cluster_ids]
    graph(dataset, centroids, 'Scatter Plot with Clusters', 'X Label', 'Y Label', colors=colors)



if __name__ == '__main__':
    # Loading iris dataset
    iris = load_iris()
    iris_data = iris.data
    iris_labels = iris.target

    kmeans(iris_data, k=3, epochs=100, labels=iris_labels, use_labels=False)
