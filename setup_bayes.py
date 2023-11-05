import numpy as np
from classification.bayes import BayesClassifier
from classification.data_generator import generate_classes, MLClass
from sklearn.model_selection import train_test_split
import py_serial 

py_serial.SERIAL_Init("COM3")

MEAN_1 = [2.5, 2.]
STD_DEV1 = [1, 2]
MEAN_2 = [1, 2]
STD_DEV2 = [.5, 1]
ml_class1 = MLClass("CLASS 1", 100, MEAN_1, STD_DEV1)
ml_class2 = MLClass("CLASS 2", 100, MEAN_2, STD_DEV2)
all_classes = [ml_class1, ml_class2]
samples, labels = generate_classes(all_classes)
train_samples, test_samples, train_labels, test_labels = train_test_split(samples, labels, test_size=0.2, random_state=42)

bayesian = BayesClassifier()
bayesian.train(train_samples, train_labels)

i = 0
while 1:

    rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
    if rqType == py_serial.MCU_WRITES:
        # INPUT -> FROM MCU TO PC
        inputs = py_serial.SERIAL_Read()
    
    elif rqType == py_serial.MCU_READS:
        # INPUT -> FROM PC TO MCU
        inputs = test_samples[i:i+1].astype(py_serial.SERIAL_GetDType(dataType))
        i = i + 1
        if i >= len(test_samples):
            i = 0
        py_serial.SERIAL_Write(inputs)
    
    pcout = bayesian.inference(np.reshape(inputs, (1, datalength)))
    rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
    if rqType == py_serial.MCU_WRITES:
        mcuout = py_serial.SERIAL_Read()
        print()
        print("Inputs : " + str(inputs))
        print("PC Output : " + str(pcout))
        print("MCU Output : " + str(mcuout))
        print()

        