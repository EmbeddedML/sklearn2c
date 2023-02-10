from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

blobs, labels = make_blobs(n_samples=100, n_features=2, centers= 2, random_state=42)
dbscan = DBSCAN(eps=2)
cluster_labels = dbscan.fit_predict(blobs)
plt.scatter(blobs[:,0], blobs[:,1], c=cluster_labels)
plt.title("DBSCAN on sample cluster data")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.show()