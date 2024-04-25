import os.path as osp
from sklearn.datasets import make_classification
from sklearn2c.clustering import (
    Kmeans
)
from sklearn.model_selection import train_test_split

samples, _ = make_classification(200, 2, n_redundant=0)
train_samples, test_samples = train_test_split(
    samples, test_size=0.2, random_state=42
)
MODELS_DIR = osp.join("models", "clustering")
CONFIG_DIR = osp.join("configs", "clustering")

kmeans = Kmeans()
kmeans_model_dir = osp.join(MODELS_DIR, "kmeans_clustering.joblib")
kmeans_config_dir = osp.join(CONFIG_DIR, "kmeans_clus_config")
kmeans.train(train_samples, save_path=kmeans_model_dir)
preds = kmeans.predict(test_samples)
kmeansian2 = Kmeans.load(kmeans_model_dir)
kmeansian2.export(kmeans_config_dir)

# dbscan = Dbscan()
# dtc_model_dir = osp.join(MODELS_DIR, "dbscan_clustering.joblib")
# dtc_config_dir = osp.join(CONFIG_DIR, "dbscan_clus_config")
# dtc.train(train_samples, train_labels, save_path=dtc_model_dir)
# dtc.predict(test_samples)
# dtc2 = DTclustering.load(dtc_model_dir)
# dtc2.export(dtc_config_dir)