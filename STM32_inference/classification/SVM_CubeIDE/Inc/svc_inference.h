#ifndef INC_SVM_INFERENCE_H_
#define INC_SVM_INFERENCE_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "SVM_config.h"

static void linear_kernel_func(float *x);
static void poly_kernel_func(float *x);
static void rbf_kernel_func(float *x);
static void calculate_ovo_scores(float *kernels, float *ovo_confs);
static void calculate_ovr_scores(float *ovo_confs, float *ovr_confs);
int svc_predict(float* input, int* output_label);


#endif /* INC_SVM_INFERENCE_H_ */
