import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import DBSCAN
from .clus_exporter import DBSCANExporter

class Dbscan:
    def __init__(self, **kwargs) -> None:
        self.clus = DBSCAN(**kwargs)
        self.means = None
        self.stds = None

    def train(self, train_samples : np.ndarray | pd.DataFrame, save_path=None):
        self.means = np.array(train_samples.mean(axis = 0))
        self.stds = np.array(train_samples.std(axis = 0))
        train_samples = (train_samples - self.means) / self.stds
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

    def export(self, filename="dbscan_clus_config"):
        dbscanWriter = DBSCANExporter(self.clus, self.means, self.stds)
        dbscanWriter.export(filename)