# Machine Learning for Embbedded Devices
sklearn2c is a tool that converts scikit-learn library classification algorithms to C code. It can be used to generate C code from trained models, which can then be used in microcontrollers or other embedded systems. The generated code can be used for real-time classification tasks, where the computational resources are limited.
## Dependencies
[scikit-learn](https://scikit-learn.org/)
## Supported Models
### Classification
- Bayes Classifier*
- Decision Trees
- KNN Classifier
- C-SVC**
  
  *: sklearn2c does not use scikit-learn `GaussianNB()`, instead it uses the following cases to compute decision function.
  **: `linear`, `poly` and `rbf` kernels are supported.
### Regression
- Linear Regression
- Polynomial Regression
- KNN
- Decision Trees
### Clustering
- kmeans
- DBSCAN

## Installation
`pip install sklearn2c`
## Usage

## Contributing
TBD
## License
[MIT](LICENSE)

