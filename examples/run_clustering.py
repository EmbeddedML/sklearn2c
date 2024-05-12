import os.path as osp
from sklearn.datasets import make_blobs
from sklearn2c.clustering import (
    Kmeans,
    Dbscan)
from sklearn.model_selection import train_test_split

samples, _ = make_blobs(200, 2, centers = 3, random_state= 42)
train_samples, test_samples = train_test_split(
    samples, test_size=0.2, random_state=42
)
MODELS_DIR = osp.join("models", "clustering")
CONFIG_DIR = osp.join("configs", "clustering")

kmeans = Kmeans()
kmeans_model_dir = osp.join(MODELS_DIR, "kmeans_clustering.joblib")
kmeans_config_dir = osp.join(CONFIG_DIR, "kmeans_clus_config")
kmeans.train(train_samples, save_path=kmeans_model_dir)
kmeans_preds = kmeans.predict(test_samples)
kmeans2 = Kmeans.load(kmeans_model_dir)
kmeans2.export(kmeans_config_dir)

dbscan = Dbscan(eps = 1)
dbscan_model_dir = osp.join(MODELS_DIR, "dbscan_clustering.joblib")
dbscan_config_dir = osp.join(CONFIG_DIR, "dbscan_clus_config")
dbscan.train(train_samples, save_path=dbscan_model_dir)
dbscan_preds =dbscan.predict(test_samples)
dbscan2 = Dbscan.load(dbscan_model_dir)
dbscan2.export(dbscan_config_dir)