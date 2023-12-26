from regression.linear_reg import LinearRegressor
from regression.polynomial_reg import PolynomialRegressor
from regression.dtr import DTRegressor
from regression.knn import KNNRegressor
from regression.reg_data_generator import generate_regression_data

train_samples, train_labels, coeff1 = generate_regression_data(100, 20, 0, rs= 9)

linear = LinearRegressor()
linear2 = LinearRegressor()
linear_save_path = "linear_reg.joblib"
linear.train(train_samples, train_labels, linear_save_path)
linear2.load(linear_save_path)
preds = linear2.inference(train_samples)
linear2.export()

poly = PolynomialRegressor(deg = 3)
poly2 = PolynomialRegressor(3)
poly_save_path = "poly_reg.joblib"
poly.train(train_samples, train_labels, poly_save_path)
poly2.load(poly_save_path)
poly2.inference(train_samples)
poly2.export()

dtr = DTRegressor()
dtr2 = DTRegressor()
dtr_save_path = "dtr.joblib"
dtr.train(train_samples, train_labels, dtr_save_path)
dtr2.load(dtr_save_path)
dtr2.inference(train_samples)
dtr2.export()

knn = KNNRegressor()
knn2 = KNNRegressor()
knn_save_path = "knn_reg.joblib"
knn.train(train_samples, train_labels, knn_save_path)
knn2.load(knn_save_path)
knn2.inference(train_samples)
knn2.export()