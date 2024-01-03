import os.path as osp
import numpy as np

def np2str(arr):
    str_arr = np.array2string(arr, separator= ",")
    str_arr = str_arr.replace("[", "{").replace("]","}")
    str_arr = str_arr.replace("'","\"")
    # str_arr = str_arr.replace("\n", ", ")
    # str_arr = str_arr.replace("  ", ", ")
    # str_arr = str_arr.replace(" ", ", ")
    return str_arr

def feats2str(arr):
    new_arr = np.array([""] * len(arr), dtype=object)
    for idx, elem in enumerate(arr):
        inter_feats = elem.split(" ")
        for intra_feats in inter_feats:
            power = intra_feats.split("^")
            feat_num = power[0].replace("x", "")
            if len(power) == 2:
                new_arr[idx] = "*".join(int(power[1]) * feat_num)
            else:
                new_arr[idx] = feat_num

    str_arr = np.array2string(new_arr, separator= ",")
    str_arr = str_arr.replace("[", "{").replace("]","}")
    str_arr = str_arr.replace("'","\"")
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

class PolynomialRegExporter(GenericExporter):
    def __init__(self, regressor, feature_names = None) -> None:
        super().__init__()
        self.regressor = regressor
        coef_start = 0
        self.offset = self.regressor.intercept_.item() if self.regressor.intercept_ else 0
        if feature_names is not None:
            coef_start = 1
            self.feature_names = feature_names[1:]
        else:
            self.feature_names = None
        
        self.coeff_str = np2str(self.regressor.coef_[0, coef_start:])

    def create_header(self):
        super().create_header()
        num_features = self.regressor.n_features_in_
        if self.feature_names is not None:
            num_features = self.regressor.n_features_in_ - 1

        self.header_str += f"#define NUM_FEATURES {num_features}\n"
        self.header_str += "extern const float COEFFS[NUM_FEATURES];\n"
        self.header_str += "extern const float OFFSET;\n"
        if self.feature_names is not None:
            self.header_str += "extern char *feature_names[NUM_FEATURES];\n"
        self.header_str += "#endif"

    def create_source(self):
        super().create_source()
        self.source_str += f"const float COEFFS[NUM_FEATURES] = {self.coeff_str};\n"
        self.source_str += f"const float OFFSET = {self.offset};\n"
        if self.feature_names is not None:
            self.source_str += f"char *feature_names[NUM_FEATURES] = {feats2str(self.feature_names)};\n"

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


class KNNExporter(GenericExporter):
    def __init__(self, regressor) -> None:
        self.regressor = regressor
        super().__init__()

    def create_header(self):
        super().create_header()
        self.header_str += f'#define NUM_NEIGHBORS {self.regressor.n_neighbors}\n'
        self.header_str += f'#define NUM_FEATURES {self.regressor.n_features_in_}\n'
        self.header_str += f'#define NUM_SAMPLES {self.regressor.n_samples_fit_}\n'
        self.header_str += 'extern const float DATA[NUM_SAMPLES][NUM_FEATURES];\n'
        self.header_str += 'extern const float DATA_VALUES[NUM_SAMPLES];\n'
        self.header_str += '#endif'

    def create_source(self):
        super().create_source()
        self.source_str += f'const float DATA[NUM_SAMPLES][NUM_FEATURES] = {np2str(self.regressor._fit_X)};\n'
        self.source_str += f'const float DATA_VALUES[NUM_SAMPLES] = {np2str(self.regressor._y)};\n'