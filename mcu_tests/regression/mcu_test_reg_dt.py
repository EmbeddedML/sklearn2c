import numpy as np
from regressors.dtr import DTRegressor
import py_serial

py_serial.SERIAL_Init("COM3")

train_samples = np.load('regression_data/reg_train_samples')
train_labels = np.load('regression_data/reg_train_labels')

dtr = DTRegressor()
dtr.load("regression_models/dtr.joblib")

i = 0
while 1:
    rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
    
    if rqType == py_serial.MCU_WRITES:
        # INPUT -> FROM MCU TO PC
        inputs = py_serial.SERIAL_Read()
    
    elif rqType == py_serial.MCU_READS:
        # INPUT -> FROM PC TO MCU
        inputs = train_samples[i:i+1].astype(py_serial.SERIAL_GetDType(dataType))
        i = i + 1
        if i >= len(train_samples):
            i = 0
        py_serial.SERIAL_Write(inputs)


    pcout = dtr.inference(inputs)
    rqType, datalength, dataType = py_serial.SERIAL_PollForRequest()
    if rqType == py_serial.MCU_WRITES:
        mcuout = py_serial.SERIAL_Read()
        print()
        print("Inputs : " + str(inputs))
        print("PC Output : " + str(pcout))
        print("MCU Output : " + str(mcuout))
        print()



