from regression.linear_reg import LinearRegressor
from regression.polynomial_reg import PolynomialRegressor
from regression.dtr import DTRegressor
from regression.knn import KNNRegressor
from regression.reg_data_generator import generate_regression_data

train_samples, train_labels, coeff1 = generate_regression_data(100, 20, 0, rs= 9)

linear = LinearRegressor()
linear.train(train_samples, train_labels)
preds = linear.inference(train_samples)
linear.export()

poly = PolynomialRegressor(deg = 3)
poly.train(train_samples, train_labels)
poly.inference(train_samples)
poly.export()

dtr = DTRegressor()
dtr.train(train_samples, train_labels)
dtr.inference(train_samples)
dtr.export()

knn = KNNRegressor()
knn.train(train_samples, train_labels)
knn.inference(train_samples)
knn.export()