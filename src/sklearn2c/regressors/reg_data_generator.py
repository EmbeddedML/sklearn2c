from sklearn.datasets import make_regression
from matplotlib import pyplot as plt

def generate_regression_data(num_samples, noise, bias, rs):
    X, y, coeffs = make_regression(n_samples= num_samples,
                                   n_features = 2,
                                   noise=noise,
                                   bias = bias,
                                   coef=True,
                                   random_state = rs)
    return X, y, coeffs

if __name__=='__main__':
    X1, y1, coeff1 = generate_regression_data(100, 20, 0, rs= 9)
    X2, y2, coeff2 = generate_regression_data(100, 10, 100, rs = 0)
    print(coeff1)# 3.6451248322167173
    print(coeff2)# 42.38550485581797
    
    plt.scatter(X1, y1)
    plt.scatter(X2, y2)
    plt.show()

