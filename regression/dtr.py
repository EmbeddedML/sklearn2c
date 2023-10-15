import joblib

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .base_regressor import BaseRegressor
from .reg_writer import DTRegressorExporter

class DTRegressor(BaseRegressor):
    def __init__(self, **kwargs):
       self.reg = DecisionTreeRegressor(**kwargs)
       super().__init__(self.reg)

    def train(self, train_samples, train_labels, save = False):
        self.reg = super().train(train_samples, train_labels)
        if save:
            joblib.dump(self.reg, 'DecisionTree_Regressor.joblib') 

    def inference(self, test_samples, test_labels = None):
        result = super().inference(test_samples, test_labels)
        return result
        
    def export(self, filename = 'dtr_config'):
        TreeWriter = DTRegressorExporter(self.reg)
        TreeWriter.export(filename)
