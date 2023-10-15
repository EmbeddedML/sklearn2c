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
            joblib.dump(self, 'bayes_classifier.joblib') 

    def inference(self, x_T, test_labels = None):
        self.result = super().inference(x_T, test_labels)
        x = np.transpose(x_T)
        discr = np.zeros(len(self.clf.classes_))
        for lbl in range(len(self.clf.classes_)):
            mu = np.expand_dims(self.clf.theta_[lbl], 1)
            sigma = self.inv_covs[lbl]
            xt_sigma = np.matmul(x_T, sigma)
            xt_sigma_x = np.matmul(xt_sigma, x) * (-1/2)
            sigma_mu = np.matmul(sigma, mu)
            sigma_mu_x = np.matmul(np.transpose(sigma_mu), x) 
            mu_sigma_mu = np.matmul(np.transpose(mu), sigma_mu) * (-1/2)
            log_det = np.log(self.det_sqrs[lbl]) * (-1/2)
            prior = np.log(self.clf.class_prior_[lbl])
            discr[lbl] = xt_sigma_x + sigma_mu_x + mu_sigma_mu + log_det + prior
        return discr
        
    def export(self, filename = 'bayes_config'):
        BayesWriter = BayesExporter(self.clf, self.inv_covs, self.det_sqrs)
        BayesWriter.export(filename)
