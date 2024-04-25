from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn2c.clustering.clus_writer import DBSCANExporter

blobs, labels = make_blobs(n_samples=100, n_features=2, centers= 2, random_state=42)
dbscan = DBSCAN(eps=2)
clustered = dbscan.fit(blobs)
DBSCANWriter = DBSCANExporter(clustered)
DBSCANWriter.export2file('clustering/DBSCAN_CubeIDE/Inc/dbscan_config.h')