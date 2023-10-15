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

class GenericExporter:
    def __init__(self) -> None:
        self.header_str = ""
        self.source_str = ""
        self.filename = ""

    def export(self, filename):
        self.filename = filename
        self.create_header()
        self.create_source()
        with open(f'{filename}.h', "w") as header_file:
            header_file.write(self.header_str)
        with open(f'{filename}.c', "w") as source_file:
            source_file.write(self.source_str)

    def create_header(self):
        self.header_str += f'#ifndef {self.filename.upper()}_H_INCLUDED\n'
        self.header_str += f'#define {self.filename.upper()}_H_INCLUDED\n'
    
    def create_source(self):
        self.source_str += f'#include "{self.filename}.h"\n'

class PolynomialRegExporter(GenericExporter):
    def __init__(self, regressor) -> None:
        super().__init__()
        self.regressor = regressor
        self.coeff_str = np2str(self.regressor.coef_)
        self.offset = self.regressor.intercept_ if self.regressor.intercept_ else 0

    def create_header(self):
        super().create_header()
        self.header_str += f"#define NUM_FEATURES {self.regressor.n_features_in_}\n"
        self.header_str += "extern const float COEFFS[NUM_FEATURES];\n"
        self.header_str += "extern const float OFFSET;\n"
        self.header_str += "#endif"

    def create_source(self):
        super().create_source()
        self.source_str += f"float COEFFS[NUM_FEATURES] = {self.coeff_str};\n"
        self.source_str += f"float OFFSET = {self.offset};\n"

class DTRegressorExporter(GenericExporter):
    def __init__(self, regressor) -> None:
        self.regressor = regressor
        super().__init__()
        self.tree = self.regressor.tree_
        self.values = np.squeeze(self.tree.value)

    def create_header(self):
        super().create_header()
        self.header_str += f"#define NUM_FEATURES {self.regressor.n_features_in_}\n"
        self.header_str += f"#define NUM_NODES {self.tree.node_count}\n"
        self.header_str += "extern const int LEFT_CHILDREN[NUM_NODES];\n"
        self.header_str += "extern const int RIGHT_CHILDREN[NUM_NODES];\n"
        self.header_str += "extern const int SPLIT_FEATURE[NUM_NODES];\n"
        self.header_str += "extern const float THRESHOLDS[NUM_NODES];\n"
        self.header_str += "extern const float VALUES[NUM_NODES];\n"
        self.header_str += "#endif"

    def create_source(self):
        super().create_source()
        self.source_str += f"const int LEFT_CHILDREN[NUM_NODES] = {np2str(self.tree.children_left)};\n"
        self.source_str += f"const int RIGHT_CHILDREN[NUM_NODES] = {np2str(self.tree.children_right)} ;\n"
        self.source_str += f"const int SPLIT_FEATURE[NUM_NODES] = {np2str(self.tree.feature)};\n"
        self.source_str += f"const float THRESHOLDS[NUM_NODES] = {np2str(self.tree.threshold)};\n"
        self.source_str += f"const float VALUES[NUM_NODES] = {np2str(self.values)};\n"