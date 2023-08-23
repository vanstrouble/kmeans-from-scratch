import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def normalize(data):
    """Normalize data"""
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    normalized_data = (data - means) / stds
    return normalized_data

def generate_prototypes(data, k, ds_size, labels=None):
    """Initializes prototypes"""
    if labels is None:
        # Random choice
        p_indices = np.random.choice(ds_size, k, replace=False)
        return data[p_indices]
    else:
        # Mean choice
        protopypes = []
        for label in np.unique(labels):
            label_data = data[labels == label]
            class_mean = np.mean(label_data, axis=0)
            protopypes.append(class_mean)

        return np.array(protopypes)

def graph(data, prototypes=None, title='', xlabel='', ylabel=''):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c='blue', label='Data Points')
    if prototypes is not None:
        plt.scatter(prototypes[:, 0], prototypes[:, 1], c='red', marker='x', label='Prototypes')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def kmeans(data, k=3, labels=None, norm=True):
    """K-means algorithm"""
    dataset = normalize(data) if norm else data

    ds_size, ds_dim = dataset.shape
    prototypes = generate_prototypes(dataset, k, ds_size, labels=labels)
    # print(f"Prototypes: \n{prototypes}")
    distances = np.zeros((ds_size, k))
    for i in range(ds_size):
        for j in range(k):
            distances[i, j] = np.linalg.norm(dataset[i] - prototypes[j])
    print(f"Distances matrix: \n{distances} \n{distances.shape}")

    cluster_ids = np.argmin(distances, axis=1)
    print(f"\nCluster IDs: \n{cluster_ids} \n{cluster_ids.shape}")

    cluster_counts = np.bincount(cluster_ids)
    for cluster_id, count in enumerate(cluster_counts):
        print(f"Cluster {cluster_id}: {count} data points")

    graph(dataset, prototypes, 'Scatter Plot with Prototypes', 'X Label', 'Y Label')
    # graph(dataset, title='Scatter Plot without Prototypes', xlabel='X Label', ylabel='Y Label')



if __name__ == '__main__':
    # Loading iris dataset
    iris = load_iris()
    iris_data = iris.data
    iris_labels = iris.target

    kmeans(iris_data, k=3,)

    # print(np.unique(iris_labels))

    # kmeans(iris_data[:, 2:])

    # ds_size, ds_dim = iris_data.shape

    # print(f"DS array type: {type(iris_data)}")  # Verify Numpy arrays
    # print(f"Data size: {ds_size}, Data dimensions: {ds_dim}")
    # print(f"\nNormalized data: \n{normalize(iris_data)}")
