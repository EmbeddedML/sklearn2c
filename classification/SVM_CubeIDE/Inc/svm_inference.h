/*
 * svm_inference.h
 *
 *  Created on: Jan 22, 2022
 *      Author: berkan
 */

#ifndef INC_SVM_INFERENCE_H_
#define INC_SVM_INFERENCE_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "svm_config.h"

float *compute_kernels(float *x);
int calculate_votes(float *kernels);

#endif /* INC_SVM_INFERENCE_H_ */
