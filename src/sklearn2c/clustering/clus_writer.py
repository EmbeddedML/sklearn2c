import numpy as np
import os.path as osp

def np2str(arr):
    str_arr = np.array2string(arr, separator= ",")
    str_arr = str_arr.replace("[", "{").replace("]","}")
    return str_arr

class GenericExporter:
    def __init__(self) -> None:
        self.header_str = ""
        self.source_str = ""
        self.filename = ""
        self.config_name = ""

    def export(self, filename):
        self.filename = filename
        self.config_name = osp.basename(self.filename)
        self.create_header()
        self.create_source()
        with open(f'{filename}.h', "w") as header_file:
            header_file.write(self.header_str)
        with open(f'{filename}.c', "w") as source_file:
            source_file.write(self.source_str)

    def create_header(self):
        self.header_str += f'#ifndef {self.config_name.upper()}_H_INCLUDED\n'
        self.header_str += f'#define {self.config_name.upper()}_H_INCLUDED\n'
    
    def create_source(self):
        self.source_str += f'#include "{self.config_name}.h"\n'

class kMeansExporter(GenericExporter):
    def __init__(self, clus) -> None:
        self.clus = clus
        super().__init__()
        self.centroids = self.clus.cluster_centers_
        self.num_clusters = self.clus.n_clusters
        self.num_features = self.clus.n_features_in_
        self.num_samples_per_cluster = np.histogram(self.clus.labels_, bins= self.num_clusters)[0]

    def create_header(self):
        super().create_header()
        self.header_str += f"#define NUM_CLUSTERS {self.num_clusters}\n"
        self.header_str += f"#define NUM_FEATURES {self.num_features}\n"
        self.header_str += "extern int num_samples_per_cluster[NUM_CLUSTERS];\n"
        self.header_str += "extern float centroids[NUM_CLUSTERS][NUM_FEATURES];\n"
        self.header_str += "#endif"
    
    def create_source(self):
        super().create_source()
        self.source_str += f"int num_samples_per_cluster[NUM_CLUSTERS] = {np2str(self.num_samples_per_cluster)};\n"
        self.source_str += f"float centroids[NUM_CLUSTERS][NUM_FEATURES] = {np2str(self.centroids)};"

class DBSCANExporter(GenericExporter):
    def __init__(self, clus) -> None:
        self.clus = clus
        super().__init__()
        self.core_points = self.clus.components_
        self.num_core_points, self.num_features = self.core_points.shape
        self.labels = self.clus.labels_
        self.eps = self.clus.eps
        self.num_clusters = len(np.unique(self.labels))

    def create_header(self):
        super().create_header()
        self.header_str += f"#define NUM_CORE_POINTS {self.num_core_points}\n"
        self.header_str += f"#define NUM_FEATURES {self.num_features}\n"
        self.header_str += f"#define NUM_CLUSTERS {self.num_clusters}\n"
        self.header_str += f"#define EPS {self.eps}\n"
        self.header_str += "extern float CORE_POINTS[NUM_CORE_POINTS][NUM_FEATURES];\n"
        self.header_str += "extern int LABELS[NUM_CORE_POINTS];\n"
        self.header_str += "#endif"
    
    def create_source(self):
        super().create_source()
        self.source_str += f"float CORE_POINTS[NUM_CORE_POINTS][NUM_FEATURES] = {np2str(self.core_points)};\n"
        self.source_str += f"int LABELS[NUM_CORE_POINTS] = {np2str(self.labels)};\n"