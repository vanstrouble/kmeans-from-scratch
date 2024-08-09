# K-Means

## Project description

This project is my own implementation of the K-means algorithm from scratch, using only Python and NumPy based on a previous version I did in the AI - LSIA class, now making a model with different datasets.

## What is K-means algorithm?

K-means clustering is an unsupervised learning algorithm used for data clustering , which groups unlabelled data points into groups or clusters.

While there are several types of clustering algorithms, including exclusive, overlapping, hierarchical and probabilistic, the k-means clustering algorithm is an example of an exclusive or "hard" clustering method. This form of clustering stipulates that a data point can exist in only one cluster. This type of cluster analysis is commonly used in data science for market segmentation, document clustering, image segmentation and image compression. The k-means algorithm is a widely used method in cluster analysis because it is efficient, effective and simple.

K-means is a centroid-based iterative clustering algorithm that divides a dataset into similar groups based on the distance between their centroids. The centroid, or centre of the cluster, is the mean or median of all points within the cluster, depending on the characteristics of the data.

<div>
    <img src="https://images.datacamp.com/image/upload/v1678462092/image7_a1777d39aa.png" width=700 alt="K-means representation image">
</div>

## Pseudocode

```plaintext
1. Initialise K random centroids.
2. Repeat until convergence or a maximum number of iterations:
    a. For each data point in the data set:
        i. Assign the data point to the nearest centroid.
    b. For each cluster:
        i. Update the centroid by calculating the mean of the points assigned to the cluster.
3. End
