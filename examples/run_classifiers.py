import os.path as osp
from sklearn.datasets import make_classification
from sklearn2c.classifiers import (
    BayesClassifier,
    KNNClassifier,
    DTClassifier,
    SVMClassifier,
)
from sklearn.model_selection import train_test_split

samples, labels = make_classification(200, 2, n_redundant=0)
train_samples, test_samples, train_labels, test_labels = train_test_split(
    samples, labels, test_size=0.2, random_state=42
)
MODELS_DIR = osp.join("models", "classification")
CONFIG_DIR = osp.join("configs", "classification")

bayesian = BayesClassifier(case=3)
bayes_model_dir = osp.join(MODELS_DIR, "bayes_classifier.joblib")
bayes_config_dir = osp.join(CONFIG_DIR, "bayes_config")
bayesian.train(train_samples, train_labels, save_path=bayes_model_dir)
preds = bayesian.predict(test_samples)
bayesian2 = BayesClassifier.load(bayes_model_dir)
bayesian2.export(bayes_config_dir)

dtc = DTClassifier()
dtc_model_dir = osp.join(MODELS_DIR, "dtc_classifier.joblib")
dtc_config_dir = osp.join(CONFIG_DIR, "dtc_config")
dtc.train(train_samples, train_labels, save_path=dtc_model_dir)
dtc.predict(test_samples)
dtc2 = DTClassifier.load(dtc_model_dir)
dtc2.export(dtc_config_dir)

knn = KNNClassifier()
knn2 = KNNClassifier()
knn_model_dir = osp.join(MODELS_DIR, "knn_classifier.joblib")
knn_config_dir = osp.join(CONFIG_DIR, "knn_classifier")
knn.train(train_samples, train_labels, save_path=knn_model_dir)
knn.predict(test_samples)
knn2 = KNNClassifier.load(knn_model_dir)
knn2.export(knn_config_dir)

svm = SVMClassifier()
svm2 = SVMClassifier()
svc_model_dir = osp.join(MODELS_DIR, "svm_classifier.joblib")
svc_config_dir = osp.join(CONFIG_DIR, "svc_config")
svm.train(train_samples, train_labels, save_path=svc_model_dir)
svm.predict(test_samples)
svm2 = SVMClassifier.load(svc_model_dir)
svm2.export(svc_config_dir)
