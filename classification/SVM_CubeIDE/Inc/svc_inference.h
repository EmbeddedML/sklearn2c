#ifndef INC_SVM_INFERENCE_H_
#define INC_SVM_INFERENCE_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "svc_config.h"

float *compute_kernels(float *x);
int calculate_ovo_scores(float *kernels, float *ovo_confs);
int calculate_ovr_scores(float *ovo_confs, float *ovr_confs);
int svc_predict(float* input, char* output_label);


#endif /* INC_SVM_INFERENCE_H_ */
