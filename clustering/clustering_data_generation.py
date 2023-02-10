from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

blobs, labels = make_blobs(n_samples=100, n_features=2, centers= 2, random_state=42)
plt.scatter(blobs)
plt.show()