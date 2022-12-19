from abc import ABC, abstractmethod
import numpy as np
import os


def np2str(arr):
    str_arr = np.array2string(arr, separator= ",")
    str_arr = str_arr.replace("[", "{").replace("]","}")
    # str_arr = str_arr.replace("\n", ", ")
    # str_arr = str_arr.replace("  ", ", ")
    # str_arr = str_arr.replace(" ", ", ")
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

class BayesExporter(GenericExporter):
    def __init__(self, bayes_classifier) -> None:
        super().__init__()
    def init_params():
        pass
    
class DTClassifierExporter(GenericExporter):
    def __init__(self, dt_classifier) -> None:
        super().__init__()

class KNNExporter(GenericExporter):
    def __init__(self, knn_classifier) -> None:
        super().__init__()

class SVMExporter(GenericExporter):
    def __init__(self, svm_classifier) -> None:
        super().__init__()

class PolynomialRegExporter(GenericExporter):
    def __init__(self, regressor) -> None:
        self.regressor = regressor
        super().__init__()
    def init_params(self):
        self.coeff_str = np2str(self.regressor.coef_)
        self.offset = self.regressor.intercept_ if self.regressor.intercept_ else 0
    def create_string(self):
        self.c_str += f"#define NUM_FEATURES {self.regressor.n_features_in_}\n"
        self.c_str += f"float COEFFS[NUM_FEATURES] = {self.coeff_str};\n"
        self.c_str += f"float OFFSET = {self.offset};\n"

class DTRegressorExporter(GenericExporter):
    def __init__(self, regressor) -> None:
        self.regressor = regressor
        super().__init__()
        self.init_params()
    def init_params(self):
        self.tree = self.regressor.tree_
        self.values = np.squeeze(self.tree.value)
    def create_string(self):
        self.c_str += f"#define NUM_FEATURES {self.regressor.n_features_in_}\n"
        self.c_str += f"#define NUM_NODES {self.tree.node_count}\n"
        self.c_str += f"const int LEFT_CHILDREN[NUM_NODES] = {np2str(self.tree.children_left)};\n"
        self.c_str += f"const int RIGHT_CHILDREN[NUM_NODES] = {np2str(self.tree.children_right)} ;\n"
        self.c_str += f"const int SPLIT_FEATURE[NUM_NODES] = {np2str(self.tree.feature)};\n"
        self.c_str += f"const float THRESHOLDS[NUM_NODES] = {np2str(self.tree.threshold)};\n"
        self.c_str += f"const float VALUES[NUM_NODES] = {np2str(self.values)};\n"