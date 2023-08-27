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

def generate_centroids(data, k, ds_size, labels=None, method='random'):
    """Initializes prototypes"""
    if method == 'random':
        p_indices = np.random.choice(ds_size, k, replace=False)
        return data[p_indices]
    elif method == 'mean':
        centroids = []
        if labels is None:
            raise ValueError('Labels must be provided dor the mean method')

        for label in np.unique(labels):
            label_data = data[labels == label]
            class_mean = np.mean(label_data, axis=0)
            centroids.append(class_mean)
        return np.array(centroids)
    else:
        raise ValueError('Invalid centroid generation method')

def accuracy(assigned_labels, true_labels, data):
    acc = np.sum(assigned_labels == true_labels) / len(true_labels)
    for index, element in enumerate(assigned_labels):
        print(f'{data[index]}, pred: {element}, expected: {true_labels[index]}')
    return acc

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

def kmeans(data, k=3, epochs=100, labels=None, norm=True, init_method='random', umbral=0.0001):
    """K-means algorithm"""
    dataset = normalize(data) if norm else data

    ds_size, ds_dim = dataset.shape
    centroids = generate_centroids(dataset, k, ds_size, labels=labels, method=init_method)
    # print(f"Prototypes: \n{prototypes}")

    converged = False
    epoch_count = 0
    while not converged and epoch_count < epochs:
        distances = np.linalg.norm(dataset[:, np.newaxis, :] - centroids, axis=2)
        cluster_ids = np.argmin(distances, axis=1)

        new_centroids = np.array([np.mean(dataset[cluster_ids == j], axis=0) for j in range(k)])
        # new_centroids = generate_centroids(dataset, k, ds_size, labels=cluster_ids, method='mean')

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

    # a = accuracy(cluster_ids, labels, dataset)
    # print(f'Accuracy: {a:.2f}')

    colors = [plt.cm.jet(float(i) / max(cluster_ids)) for i in cluster_ids]
    graph(dataset, centroids, 'Scatter Plot with Clusters', 'X Label', 'Y Label', colors=colors)



if __name__ == '__main__':
    print('Welcome to Kmeans')
    setup = int(input('Select a dataset [1. Iris, 2. Randomized]: '))

    if setup == 1:
        # Loading iris dataset
        iris = load_iris()
        iris_data = iris.data
        iris_labels = iris.target
        kmeans(iris_data, k=3, epochs=100, norm=False, labels=iris_labels, init_method='random')

    else:
        random_points = np.random.randint(0, 100, (100, 2))
        kmeans(random_points, k=3, epochs=200, )
