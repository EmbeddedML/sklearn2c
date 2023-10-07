from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def arr2str(arr):
    if type(arr) in (list, tuple):
        arr = np.array(arr)
    str_arr = np.array2string(arr, separator= ",")
    str_arr = str_arr.replace("[", "{").replace("]","}")
    str_arr = str_arr.replace("'","\"")
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
        self.header_str += '#define BAYES_CONFIG_H_INCLUDED\n'
    
    def create_source(self):
        self.source_str += f'#include "{self.filename}.h"\n'

class BayesExporter(GenericExporter):
    def __init__(self, bayes_classifier:DecisionTreeClassifier, inv_covs, sqrt_dets) -> None:
        super().__init__()
        self.clf = bayes_classifier
        self.num_classes = len(self.clf.class_count_)
        self.sqrt_dets = sqrt_dets
        self.inv_covs = np.array(inv_covs).reshape(self.num_classes, -1)

    def create_header(self):
        super().create_header()
        self.header_str += f"#define NUM_CLASSES {self.num_classes}\n"
        self.header_str += f"#define NUM_FEATURES {self.clf.n_features_in_}\n"
        self.header_str += "extern const char *LABELS[NUM_CLASSES];\n"
        self.header_str += "extern float MEANS[NUM_CLASSES][NUM_FEATURES];\n"
        self.header_str += "extern float INV_COVS[NUM_CLASSES][NUM_FEATURES * NUM_FEATURES];\n"
        self.header_str += "extern const float DETS[NUM_CLASSES];\n"
        self.header_str += "extern const float CLASS_PRIORS[NUM_CLASSES];\n"
        self.header_str += '#endif\n'
    
    def create_source(self):
        super().create_source()
        self.source_str += f'const char *LABELS[NUM_CLASSES] = {arr2str(self.clf.classes_)};\n'
        self.source_str += f'float MEANS[NUM_CLASSES][NUM_FEATURES] = {arr2str(self.clf.theta_)};\n'
        self.source_str += f'float INV_COVS[NUM_CLASSES][NUM_FEATURES * NUM_FEATURES] = {arr2str(self.inv_covs)};\n'
        self.source_str += f'const float DETS[NUM_CLASSES] = {arr2str(self.sqrt_dets)};\n'
        self.source_str += f'const float CLASS_PRIORS[NUM_CLASSES] = {arr2str(self.clf.class_prior_)};\n'
    
class DTClassifierExporter(GenericExporter):
    def __init__(self, dt_classifier) -> None:
        self.clf = dt_classifier
        super().__init__()
        self.tree = dt_classifier.tree_
        self.values = np.squeeze(self.tree.value)

    def create_header(self):
        super().create_header()
        self.header_str += f"#define NUM_NODES {self.tree.node_count}\n"
        self.header_str += f"#define NUM_FEATURES {self.clf.n_features_in_}\n"
        self.header_str += "extern const int LEFT_CHILDREN[NUM_NODES];\n"
        self.header_str += "extern const int RIGHT_CHILDREN[NUM_NODES];\n"
        self.header_str += "extern const int SPLIT_FEATURE[NUM_CLASSES];\n"
        self.header_str += "extern const float THRESHOLDS[NUM_CLASSES];\n"
        self.header_str += "extern const float VALUES[NUM_NODES];\n"
        self.header_str += '#endif\n'

    def create_source(self):
        super().create_source()
        self.source_str += f"const int LEFT_CHILDREN[NUM_NODES] = {arr2str(self.tree.children_left)};\n"
        self.source_str += f"const int RIGHT_CHILDREN[NUM_NODES] = {arr2str(self.tree.children_right)} ;\n"
        self.source_str += f"const int SPLIT_FEATURE[NUM_NODES] = {arr2str(self.tree.feature)};\n"
        self.source_str += f"const float THRESHOLDS[NUM_NODES] = {arr2str(self.tree.threshold)};\n"
        self.source_str += f"const float VALUES[NUM_NODES] = {arr2str(self.values)};\n"
    

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
        self.source_str += '#endif'
    

class SVMExporter(GenericExporter):
    def __init__(self, svm_classifier) -> None:
        super().__init__()
        self.clf = svm_classifier
        self.num_classes = len(self.clf.classes_)
        self.num_intercept = self.num_classes * (self.num_classes - 1) / 2
    
    def create_header(self):
        super().create_header()
        self.header_str += f'#define NUM_CLASSES {self.num_classes}\n'
        self.header_str += f'#define NUM_INTERCEPTS {self.num_intercept}\n'
        self.header_str += f'#define NUM_FEATURES {self.clf.n_features_in_}\n'
        self.header_str += f'#define NUM_SV {np.sum(self.clf.n_support_)}\n'
        self.header_str += 'extern const char *LABELS[NUM_CLASSES];\n'
        self.header_str += 'extern const float coeffs[NUM_FEATURES][NUM_SV];\n'
        self.header_str += 'extern const float SV[NUM_SV][NUM_FEATURES];\n'
        self.header_str += 'extern const float intercepts[NUM_CLASSES];\n'
        self.header_str += 'extern const float w_sum[NUM_CLASSES + 1];\n'
        self.header_str += 'extern const float svm_gamma;\n'
        self.header_str += '#endif'
    
    def create_source(self):
        super().create_source()
        self.source_str += f'const char *LABELS[NUM_CLASSES] = {arr2str(self.clf.classes_)};\n'
        self.source_str += f'float coeffs[NUM_FEATURES - 1][NUM_SV] = {arr2str(self.clf.dual_coef_)};\n'
        self.source_str += f'float SV[NUM_SV][NUM_FEATURES] = {arr2str(self.clf.support_vectors_)};\n'
        self.source_str += f'float intercepts[NUM_INTERCEPTS] = {arr2str(self.clf.intercept_)};\n'
        self.source_str += f'float w_sum[NUM_CLASSES + 1] = {arr2str(np.cumsum(self.clf.n_support_))};\n'
        self.source_str += f'float svm_gamma = {self.clf._gamma};\n'

