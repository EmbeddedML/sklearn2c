# Machine Learning for Embbedded Devices
sklearn2c is a tool that converts scikit-learn library classification algorithms to C code. It can be used to generate C code from trained models, which can then be used in microcontrollers or other embedded systems. The generated code can be used for real-time classification tasks, where the computational resources are limited.

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
You can install the library via pip either using:

`pip install sklearn2c`

or

`pip install git+git@github.com:EmbeddedML/sklearn2c.git`

Alternatively, you can install conda package:

`conda install sklearn2c` or `mamba install sklearn2c`

## Usage

Please check `examples` directory under this repository. For example, decision tree classifier is created as follows:
- `train` method trains the model and optionally saves the model file to `save_path`. This method is totally compatible with scikit-learn library. 
- `predict` method runs the model on the given data.
- static method `load` loads the model from saved path.
- `export` method generates model parameters as C functions.

```
dtc = DTClassifier()
dtc.train(train_samples, train_labels, save_path="<path/to/model>")
dtc.predict(test_samples)
dtc2 = DTClassifier.load(dtc_model_dir)
dtc2.export("<path/to/config_dir>")
```
For inference on C(specifically for STM32 boards), you can take a look at `STM32_inference` directory for the corresponding model.

## License
[MIT](LICENSE)

