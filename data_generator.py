import numpy as np

np.random.seed(42)  # For reproducibility
MEAN_1 = [2.5, 2]
MEAN_2 = [5, 1]
STD_DEV1 = [1, 2]
STD_DEV2 = [0.5, 1]

# Define number of samples for each class
CLS1_SAMPLE_SIZE = 100
CLS2_SAMPLE_SIZE = 100

# Generate samples from random distribution
class_1 = np.random.normal(MEAN_1, STD_DEV1, (CLS1_SAMPLE_SIZE, 2))
class_2 = np.random.normal(MEAN_2, STD_DEV2, (CLS2_SAMPLE_SIZE, 2))

# Concatenate all the samples and generate their corresponding labels
X = np.concatenate((class_1, class_2), axis=0)
y = np.array(['CLASS 1'] * CLS1_SAMPLE_SIZE + ['CLASS 2'] *
             CLS2_SAMPLE_SIZE)
