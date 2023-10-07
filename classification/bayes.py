import joblib
from math import sqrt

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import empirical_covariance

from .base_classifier import BaseClassifier
from .clf_writer import BayesExporter

class BayesClassifier(BaseClassifier):
    def __init__(self, **kwargs):
       self.clf = GaussianNB(**kwargs)
       super().__init__(self.clf)
       self.inv_covs = []
       self.det_sqrs = []

    def train(self, train_samples, train_labels, save = False):
        self.clf = super().train(train_samples, train_labels)
        # list of dict for inverse covariance matrix and determinant
        LABELS = self.clf.classes_
        for label in LABELS:
            cov_matrix = empirical_covariance(train_samples[train_labels == label])
            inv_cov = np.linalg.inv(cov_matrix)
            sqrt_det = sqrt(np.linalg.det(inv_cov))
            self.inv_covs.append(inv_cov)
            self.det_sqrs.append(sqrt_det)
        if save:
            joblib.dump(self.clf, 'bayes_classifier.joblib') 

    def inference(self, test_samples, test_labels = None):
        self.result = super().inference(test_samples, test_labels)
        
    def export(self, filename = 'bayes_config'):
        BayesWriter = BayesExporter(self.clf, self.inv_covs, self.det_sqrs)
        BayesWriter.export(filename)
