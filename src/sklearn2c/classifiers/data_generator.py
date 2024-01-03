import numpy as np

class MLClass:
    def __init__(self, name, num_samples, mu, sigma) -> None:
        self.name = name
        self.mu = mu
        self.sigma = sigma
        self.num_samples = num_samples
        self.num_features = len(self.mu)
        self.check_vars()
    def check_vars(self):
        assert len(self.mu) == len(self.sigma)
    def generate_gaussian(self, seed:int=None):
        # Generate samples from random distribution
        if seed:
            np.random.seed(seed)  # For reproducibility
        return np.random.normal(self.mu, self.sigma, (self.num_samples, self.num_features))

def generate_classes(class_list):
    assert len(class_list) > 1, "Number of classes must be greater than 1"
    X = None
    y = None
    for current_class in class_list:
        # Concatenate all the samples and generate their corresponding labels
        cur_distrib = current_class.generate_gaussian(seed=42)
        cur_label = [current_class.name] * current_class.num_samples
        X = cur_distrib if X is None else np.vstack((X, cur_distrib))
        y = cur_label if y is None else np.hstack((y, cur_label))
    return X, y

if __name__=="__main__":
    # Define number of samples for each class
    MEAN_1 = [2.5, 2.]
    STD_DEV1 = [1, 2]
    MEAN_2 = [1, 2]
    STD_DEV2 = [.5, 1]
    ml_class1 = MLClass("CLASS 1", 100, MEAN_1, STD_DEV1)
    ml_class2 = MLClass("CLASS 2", 100, MEAN_2, STD_DEV2)
    all_classes = [ml_class1, ml_class2]
    samples, labels = generate_classes(all_classes)
    print(samples.shape , labels.shape)
