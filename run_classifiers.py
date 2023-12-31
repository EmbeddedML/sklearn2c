import os.path as osp
from classification import (
    generate_classes,
    MLClass,
    BayesClassifier,
    KNNClassifier,
    DTClassifier,
    SVMClassifier,
)
from sklearn.model_selection import train_test_split

MEAN_1 = [2.5, 2.0]
STD_DEV1 = [1, 2]
MEAN_2 = [1, 2]
STD_DEV2 = [0.5, 1]
ml_class1 = MLClass("CLASS 1", 100, MEAN_1, STD_DEV1)
ml_class2 = MLClass("CLASS 2", 100, MEAN_2, STD_DEV2)
all_classes = [ml_class1, ml_class2]
samples, labels = generate_classes(all_classes)
train_samples, test_samples, train_labels, test_labels = train_test_split(
    samples, labels, test_size=0.2, random_state=42
)
MODELS_DIR = osp.join("models", "classification")
CONFIG_DIR = osp.join("configs", "classification")

bayesian = BayesClassifier(case=3)
bayesian2 = BayesClassifier(case=3)
bayes_model_dir = osp.join(MODELS_DIR, "bayes_classifier.joblib")
bayes_config_dir = osp.join(CONFIG_DIR, "bayes_config")
bayesian.train(train_samples, train_labels, save_path=bayes_model_dir)
preds = bayesian.inference(test_samples[0:1])
bayesian2.load(bayes_model_dir)
bayesian2.export(bayes_config_dir)

dtc = DTClassifier()
dtc2 = DTClassifier()
dtc_model_dir = osp.join(MODELS_DIR, "dtc_classifier.joblib")
dtc_config_dir = osp.join(CONFIG_DIR, "dtc_config")
dtc.train(train_samples, train_labels, save_path=dtc_model_dir)
dtc.inference(test_samples)
dtc2.load(dtc_model_dir)
dtc2.export(dtc_config_dir)

knn = KNNClassifier()
knn2 = KNNClassifier()
knn_model_dir = osp.join(MODELS_DIR, "knn_classifier.joblib")
knn_config_dir = osp.join(CONFIG_DIR, "knn_classifier")
knn.train(train_samples, train_labels, save_path=knn_model_dir)
knn.inference(test_samples)
knn2.load(knn_model_dir)
knn2.export(knn_config_dir)

svm = SVMClassifier()
svm2 = SVMClassifier()
svc_model_dir = osp.join(MODELS_DIR, "svm_classifier.joblib")
svc_config_dir = osp.join(CONFIG_DIR, "svc_config")
svm.train(train_samples, train_labels, save_path=svc_model_dir)
svm.inference(test_samples)
svm2.load(svc_model_dir)
svm2.export(svc_config_dir)
