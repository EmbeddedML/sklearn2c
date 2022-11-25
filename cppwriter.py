from abc import ABC, abstractmethod
import numpy as np
import os


def np2str(arr):
    str_arr = np.array2string(arr)
    str_arr = str_arr.replace("[", "{").replace("]","}")
    str_arr = str_arr.replace("  ", ", ")
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

class DTRegressorExporter():
    def __init__(self) -> None:
        super().__init__()
    def init_params(self):
        pass
    def create_string(self):
        self.c_str += f"#define NUM_FEATURES {0}\n"
        self.c_str += f"#define NUM_NODES {0}\n"
        self.c_str += f"int LEFT_CHILDREN[NUM_NODES] = {self.left_children};\n"
        self.c_str += f"int RIGHT_CHILDREN[NUM_NODES] = {self.right_children} ;\n"
        self.c_str += f"int SPLIT_FEATURE[NUM_NODES] = {self.split_features};\n"
        self.c_str += f"float THRESHOLDS[NUM_NODES] = {self.thresholds};\n"
        self.c_str += f"int COEFFS[NUM_NODES] = {self.coeffs};\n"


        
