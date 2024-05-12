import joblib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import DBSCAN
from .clus_writer import DBSCANExporter


class Dbscan:
    def __init__(self, **kwargs) -> None:
        self.clus = DBSCAN(**kwargs)

    def train(self, train_samples, save_path=None):
        self.clus.fit(train_samples, save_path)
        self.clus.labels_= self.clus.labels_[self.clus.core_sample_indices_]
        if save_path:
            joblib.dump(self, save_path)

    def predict(self, test_samples):
        result = np.full(len(test_samples), -1)
        dist_idx, distances = pairwise_distances_argmin_min(test_samples, self.clus.components_)
        cand_core_points = np.where(distances < self.clus.eps)[0]
        comp_idx = self.clus.labels_[dist_idx[cand_core_points]]
        result[cand_core_points] = comp_idx
        return result

    @staticmethod
    def load(filename: str) -> "Dbscan":
        with open(filename, "rb") as joblib_file:
            model = joblib.load(joblib_file)
        if not isinstance(model, Dbscan):
            raise TypeError(
                f"Expected an object of type Dbscan, but got {type(model)} instead."
            )
        return model

    def export(self, filename="kmeans_clus_config"):
        kMeansWriter = DBSCANExporter(self.clus)
        kMeansWriter.export(filename)