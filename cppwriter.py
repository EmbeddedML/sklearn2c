from abc import ABC, abstractmethod
import numpy
import os

class GenericExporter(ABC):
    def __init__(self) -> None:
        self.str = ""
        self.init_params()
    def export2file(self, filename):
        with open(filename, "w") as c_file:
            c_file.write(self.str)
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
    
class DecisionTreeExporter(GenericExporter):
    def __init__(self, dt_classifier) -> None:
        super().__init__()

class KNNExporter(GenericExporter):
    def __init__(self, knn_classifier) -> None:
        super().__init__()

class SVMExporter(GenericExporter):
    def __init__(self, svm_classifier) -> None:
        super().__init__()
        
