from abc import ABC, abstractmethod
import numpy as np
import os

def np2str(arr):
    str_arr = np.array2string(arr, separator= ",")
    str_arr = str_arr.replace("[", "{").replace("]","}")
    return str_arr

class GenericExporter(ABC):
    def __init__(self) -> None:
        self.c_str = ""
        self.init_params()
    def export2file(self, filename):
        self.create_string()
        with open(filename, "w") as c_file:
            c_file.write(self.c_str)
    @abstractmethod
    def init_params(self):
        pass
    @abstractmethod
    def create_string(self):
        pass

class kMeansExporter(GenericExporter):
    def __init__(self, estimator) -> None:
        self.estimator = estimator
        super().__init__()
        self.init_params()
    def init_params(self):
        self.centroids = self.estimator.cluster_centers_
        self.num_clusters = self.estimator.n_clusters
        self.num_features = self.estimator.n_features_in_
        self.num_samples_per_cluster = np.histogram(self.estimator.labels_, bins= self.num_clusters)[0]
    def create_string(self):
        self.c_str += f"#define NUM_CLUSTERS {self.num_clusters}\n"
        self.c_str += f"#define NUM_FEATURES {self.num_features}\n"
        self.c_str += f"int num_samples_per_cluster[NUM_CLASSES] = {np2str(self.num_samples_per_cluster)};\n"
        self.c_str += f"float centroids[NUM_CLUSTERS][NUM_FEATURES] = {np2str(self.centroids)};\n"

class DBSCANExporter(GenericExporter):
    def __init__(self, estimator) -> None:
        self.estimator = estimator
        super().__init__()
        self.init_params()
    def init_params(self):
        self.samples = self.estimator.components_
        self.num_samples, self.num_features = self.samples.shape
        self.labels = self.estimator.labels_
        self.eps = self.estimator.eps
        self.min_samples = self.estimator.min_samples
        self.num_clusters = len(np.unique(self.labels))
    def create_string(self):
        self.c_str += f"#define NUM_SAMPLES {self.num_samples}\n"
        self.c_str += f"#define NUM_FEATURES {self.num_features}\n"
        self.c_str += f"#define NUM_CLUSTERS {self.num_clusters}\n"
        self.c_str += f"#define EPS {self.eps}\n"
        self.c_str += f"#define MIN_SAMPLES {self.estimator.min_samples}\n"
        self.c_str += f"float samples[NUM_SAMPLES][NUM_FEATURES] = {np2str(self.samples)};\n"
        self.c_str += f"int labels[NUM_SAMPLES] = {np2str(self.labels)};\n"
