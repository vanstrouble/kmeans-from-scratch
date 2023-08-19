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


def kmeans(data, k=3, labels=None, norm=True):
    """K-means algorithm"""
    if norm:
        dataset = normalize(data)
    else:
        dataset = data

    ds_size, ds_dim = dataset.shape
    prototypes = generate_prototypes(dataset, k, ds_size, labels=labels)
    print(f"Prototypes: \n{prototypes}")



if __name__ == '__main__':
    # Loading iris dataset
    iris = load_iris()
    iris_data = iris.data
    iris_labels = iris.target

    kmeans(iris_data, k=3, labels=iris_labels)

    # print(np.unique(iris_labels))

    # kmeans(iris_data[:, 2:])

    # ds_size, ds_dim = iris_data.shape

    # print(f"DS array type: {type(iris_data)}")  # Verify Numpy arrays
    # print(f"Data size: {ds_size}, Data dimensions: {ds_dim}")
    # print(f"\nNormalized data: \n{normalize(iris_data)}")
