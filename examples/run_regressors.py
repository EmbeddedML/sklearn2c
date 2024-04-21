import os.path as osp
from sklearn.datasets import make_regression
from sklearn2c.regressors import (
    LinearRegressor,
    PolynomialRegressor,
    DTRegressor,
    KNNRegressor)

train_samples, train_labels = make_regression(100, 2)

MODELS_DIR = osp.join("models", "regression")
CONFIG_DIR = osp.join("configs", "regression")

linear = LinearRegressor()
linear_save_dir = osp.join(MODELS_DIR, "linear_reg.joblib")
linear_config_dir = osp.join(CONFIG_DIR, "linear_reg")
linear.train(train_samples, train_labels, linear_save_dir)
linear2 = LinearRegressor.load(linear_save_dir)
linear2.predict(train_samples)
linear2.export(linear_config_dir)

poly = PolynomialRegressor(deg=3)
poly2 = PolynomialRegressor(3)
poly_save_dir = osp.join(MODELS_DIR, "poly_reg.joblib")
poly_config_dir = osp.join(CONFIG_DIR, "poly_reg")
poly.train(train_samples, train_labels, poly_save_dir)
poly2= PolynomialRegressor.load(poly_save_dir)
poly2.predict(train_samples)
poly2.export(poly_config_dir)

dtr = DTRegressor()
dtr_save_dir = osp.join(MODELS_DIR, "dtr.joblib")
dtr_config_dir = osp.join(CONFIG_DIR, "dtr_reg")
dtr.train(train_samples, train_labels, dtr_save_dir)
dtr2= DTRegressor.load(dtr_save_dir)
dtr2.predict(train_samples)
dtr2.export(dtr_config_dir)

knn = KNNRegressor()
knn_save_dir = osp.join(MODELS_DIR, "knn_reg.joblib")
knn_config_dir = osp.join(CONFIG_DIR, "knn_reg")
knn.train(train_samples, train_labels, knn_save_dir)
knn2= KNNRegressor.load(knn_save_dir)
knn2.predict(train_samples)
knn2.export(knn_config_dir)
