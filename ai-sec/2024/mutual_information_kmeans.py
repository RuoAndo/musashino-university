from sklearn import datasets
from sklearn import cluster

blobs, ground_truth = datasets.make_blobs(1000, centers=3,cluster_std=1.75)

from sklearn import datasets
from sklearn import cluster

%matplotlib inline
import matplotlib.pyplot as plt
 
f, ax = plt.subplots(figsize=(7, 5))
colors = ['r', 'g', 'b']
for i in range(3):
     p = blobs[ground_truth == i]
     ax.scatter(p[:,0], p[:,1], c=colors[i],
     label="Cluster {}".format(i))
ax.set_title("Cluster with Ground Truth")
ax.legend()

kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(blobs)

f, ax = plt.subplots(figsize=(7, 5))
colors = ['r', 'g', 'b']
for i in range(3):
     p = blobs[kmeans.labels_ == i]
     ax.scatter(p[:,0], p[:,1], c=colors[i],
     label="Cluster {}".format(i))
ax.set_title("Result of Kmeans")
ax.legend()

from sklearn import metrics
metrics.normalized_mutual_info_score(ground_truth, kmeans.labels_)