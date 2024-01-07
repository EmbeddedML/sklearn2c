import os.path as osp
import numpy as np

def arr2str(arr):
    if type(arr) in (list, tuple):
        arr = np.array(arr)
    str_arr = np.array2string(arr, separator= ",")
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

class BayesExporter(GenericExporter):
    def __init__(self, bayes_classifier) -> None:
        super().__init__()
        self.clf = bayes_classifier
        self.num_classes = len(self.clf.classes)

    def create_header(self):
        super().create_header()
        self.header_str += f"#define NUM_CLASSES {self.num_classes}\n"
        self.header_str += f"#define NUM_FEATURES {self.clf.num_features}\n"
        self.header_str += f"#define CASE {self.clf.case}\n"
        self.header_str += "extern float MEANS[NUM_CLASSES][NUM_FEATURES];\n"
        self.header_str += "extern const float CLASS_PRIORS[NUM_CLASSES];\n"
        if self.clf.case == 1:
            self.header_str += "extern const float sigma_sq;\n"
        elif self.clf.case ==  2:
            self.header_str += "extern const float INV_COV[NUM_FEATURES][NUM_FEATURES];\n"
        else:
            self.header_str += "extern const float INV_COVS[NUM_CLASSES][NUM_FEATURES][NUM_FEATURES];\n"
            self.header_str += "extern const float DETS[NUM_CLASSES];\n"
        self.header_str += '#endif\n'
    
    def create_source(self):
        super().create_source()
        self.source_str += f'float MEANS[NUM_CLASSES][NUM_FEATURES] = {arr2str(self.clf.means)};\n'
        self.source_str += f'const float CLASS_PRIORS[NUM_CLASSES] = {arr2str(self.clf.priors)};\n'
        if self.clf.case == 1:
            self.source_str += f'const float sigma_sq = {self.clf.sigma_sq};\n'
        elif self.clf.case == 2:
            self.source_str += f'const float INV_COV[NUM_FEATURES][NUM_FEATURES] = {arr2str(self.clf.inv_cov)};\n'
        elif self.clf.case == 3: 
            self.source_str += f'const float INV_COVS[NUM_CLASSES][NUM_FEATURES][NUM_FEATURES] = {arr2str(self.clf.inv_covs)};\n'
            self.source_str += f'const float DETS[NUM_CLASSES] = {arr2str(self.clf.det_sqrs)};\n'
    
class DTClassifierExporter(GenericExporter):
    def __init__(self, dt_classifier) -> None:
        self.clf = dt_classifier
        super().__init__()
        self.tree = dt_classifier.tree_
        self.values = np.squeeze(self.tree.value).astype(int)

    def create_header(self):
        super().create_header()
        self.header_str += f"#define NUM_NODES {self.tree.node_count}\n"
        self.header_str += f"#define NUM_FEATURES {self.clf.n_features_in_}\n"
        self.header_str += f"#define NUM_CLASSES {len(self.clf.classes_)}\n"
        self.header_str += "extern const int LEFT_CHILDREN[NUM_NODES];\n"
        self.header_str += "extern const int RIGHT_CHILDREN[NUM_NODES];\n"
        self.header_str += "extern const int SPLIT_FEATURE[NUM_NODES];\n"
        self.header_str += "extern const float THRESHOLDS[NUM_NODES];\n"
        self.header_str += "extern const int VALUES[NUM_NODES][NUM_CLASSES];\n"
        self.header_str += '#endif\n'

    def create_source(self):
        super().create_source()
        self.source_str += f"const int LEFT_CHILDREN[NUM_NODES] = {arr2str(self.tree.children_left)};\n"
        self.source_str += f"const int RIGHT_CHILDREN[NUM_NODES] = {arr2str(self.tree.children_right)} ;\n"
        self.source_str += f"const int SPLIT_FEATURE[NUM_NODES] = {arr2str(self.tree.feature)};\n"
        self.source_str += f"const float THRESHOLDS[NUM_NODES] = {arr2str(self.tree.threshold)};\n"
        self.source_str += f"const int VALUES[NUM_NODES][NUM_CLASSES] = {arr2str(self.values)};\n"
    

class KNNExporter(GenericExporter):
    def __init__(self, knn_classifier) -> None:
        self.clf = knn_classifier
        super().__init__()

    
    def create_header(self):
        super().create_header()
        self.header_str += f'#define NUM_CLASSES {len(self.clf.classes_)}\n'
        self.header_str += f'#define NUM_NEIGHBORS {self.clf.n_neighbors}\n'
        self.header_str += f'#define NUM_FEATURES {self.clf.n_features_in_}\n'
        self.header_str += f'#define NUM_SAMPLES {self.clf.n_samples_fit_}\n'
        self.header_str += 'extern char* LABELS[NUM_CLASSES];\n'
        self.header_str += 'extern const float DATA[NUM_SAMPLES][NUM_FEATURES];\n'
        self.header_str += 'extern const int DATA_LABELS[NUM_SAMPLES];\n'
        self.header_str += '#endif'

    def create_source(self):
        super().create_source()
        self.source_str += f'char* LABELS[NUM_CLASSES] = {arr2str(self.clf.classes_)};\n'
        self.source_str += f'const float DATA[NUM_SAMPLES][NUM_FEATURES] = {arr2str(self.clf._fit_X)};\n'
        self.source_str += f'const int DATA_LABELS[NUM_SAMPLES] = {arr2str(self.clf._y)};\n'
    

class SVMExporter(GenericExporter):
    def __init__(self, svm_classifier) -> None:
        super().__init__()
        self.clf = svm_classifier
        self.num_classes = len(self.clf.classes_)
        self.num_intercept = self.num_classes * (self.num_classes - 1) // 2
        self.w_sum_arr = np.append([0], np.cumsum(self.clf.n_support_))
    
    def create_header(self):
        super().create_header()
        self.header_str += f'#define NUM_CLASSES {self.num_classes}\n'
        self.header_str += f'#define NUM_INTERCEPTS {self.num_intercept}\n'
        self.header_str += f'#define NUM_FEATURES {self.clf.n_features_in_}\n'
        self.header_str += f'#define NUM_SV {np.sum(self.clf.n_support_)}\n'
        self.header_str += 'extern const char *LABELS[NUM_CLASSES];\n'
        self.header_str += 'extern const float coeffs[NUM_CLASSES - 1][NUM_SV];\n'
        self.header_str += 'extern const float SV[NUM_SV][NUM_FEATURES];\n'
        self.header_str += 'extern const float intercepts[NUM_INTERCEPTS];\n'
        self.header_str += 'extern const float w_sum[NUM_CLASSES + 1];\n'
        self.header_str += 'extern const float svm_gamma;\n'
        self.header_str += '#endif'
    
    def create_source(self):
        super().create_source()
        self.source_str += f'const char *LABELS[NUM_CLASSES] = {arr2str(self.clf.classes_)};\n'
        self.source_str += f'const float coeffs[NUM_CLASSES - 1][NUM_SV] = {arr2str(self.clf.dual_coef_)};\n'
        self.source_str += f'const float SV[NUM_SV][NUM_FEATURES] = {arr2str(self.clf.support_vectors_)};\n'
        self.source_str += f'const float intercepts[NUM_INTERCEPTS] = {arr2str(self.clf.intercept_)};\n'
        self.source_str += f'const float w_sum[NUM_CLASSES + 1] = {arr2str(self.w_sum_arr)};\n'
        self.source_str += f'const float svm_gamma = {self.clf._gamma};\n'

