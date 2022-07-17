from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.covariance import empirical_covariance
from data_generator import generate_classes
from cppwriter import BayesExporter

X, y = generate_classes(False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
gnb = GaussianNB()
fit_func = gnb.fit(X_train, y_train)
# list of dict for inverse covariance matrix and determinant
cov_props = [dict(), dict()]
LABELS = fit_func.classes_
for label in LABELS:
    cov_matrix = empirical_covariance(X_train[y_train == label])
    inv_cov = np.linalg.inv(cov_matrix)
    sqrt_det = sqrt(np.linalg.det(inv_cov))
    cov_props[0][label] = inv_cov
    cov_props[1][label] = sqrt_det
BayesWriter = BayesExporter(fit_func, cov_props)
BayesWriter.export2file('bayes_config.c')
