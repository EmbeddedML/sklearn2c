import joblib
from math import sqrt

import numpy as np
from sklearn.covariance import empirical_covariance
from .clf_writer import BayesExporter

class BayesClassifier():
    def __init__(self, case = 3):
        if case not in (1,2,3): 
            raise ValueError("case must be either 1,2 or 3")
        self.case = case
        self.sigma_sq = None
        self.num_features = None
        self.classes = None
        self.priors = None
        self.means = []
        self.inv_covs = []
        self.det_sqrs = []

    def train(self, train_samples, train_labels, save_path):
        # list of dict for inverse covariance matrix and determinant
        self.classes, counts = np.unique(train_labels, return_counts=True)
        self.priors = counts / np.sum(counts)
        self.num_features = train_samples.shape[1]
        if self.case == 1:
            self.sigma_sq = np.var(train_samples)
        elif self.case == 2:
            cov_matrix = empirical_covariance(train_samples)
            self.inv_cov = np.linalg.inv(cov_matrix)

        for label in self.classes:
            self.means.append(np.mean(train_samples[train_labels == label], axis=0))
            if self.case == 3:
                cov_matrix = empirical_covariance(train_samples[train_labels == label])
                inv_cov = np.linalg.inv(cov_matrix)
                sqrt_det = sqrt(np.linalg.det(inv_cov))
                self.inv_covs.append(inv_cov)
                self.det_sqrs.append(sqrt_det)
        if save_path:
            joblib.dump(self, save_path)

    def load(self, filename):
        with open(filename, "rb") as joblib_file:
            saved_model = joblib.load(joblib_file)
        self.__dict__.update(saved_model.__dict__)
        
    def inference(self, X):
        num_classes = len(self.classes)
        probs = np.zeros(num_classes)

        for label in range(num_classes):
            mu_i = self.means[label]
            p = self.priors[label]
            zero_mean = X - mu_i
            if self.case == 1: 
                prod = zero_mean @ np.transpose(zero_mean)
                probs[label] = np.log(p) - prod / (2 * self.sigma_sq)
            elif self.case == 2:
                prod = zero_mean @ self.inv_cov @np.transpose(zero_mean)
                probs[label] = np.log(p) - prod * 0.5
            elif self.case == 3:
                inv_cov_i = self.inv_covs[label]
                W_i = -0.5 * inv_cov_i
                w_i = inv_cov_i @ mu_i
                w_i0 = -0.5 * mu_i @ inv_cov_i @ mu_i - 0.5 * self.det_sqrs[label] + np.log(p)
                first_term = X @ W_i @ X.T
                second_term = X @ w_i
                probs[label] = first_term + second_term + w_i0
        
        return probs
        
    def export(self, filename = 'bayes_config'):
        BayesWriter = BayesExporter(self)
        BayesWriter.export(filename)
