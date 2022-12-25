from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

blobs, labels = make_blobs(n_samples=100, n_features=2, centers= 2, random_state=42)
kmeans = KMeans(n_clusters = 2)
clustered = kmeans.fit_transform(blobs)


