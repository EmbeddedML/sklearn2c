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

float *compute_kernels(float *x)
{
    float *kernels = malloc(NUM_SV * sizeof(float));
    for (int sv_idx = 0; sv_idx < NUM_SV; sv_idx++)
    {
        float kernel = 0.0;
        for (int feature_idx = 0; feature_idx < NUM_FEATURES; feature_idx++)
        {
            kernel += pow(x[feature_idx] - SV[sv_idx][feature_idx], 2);
        }
        kernels[sv_idx] = exp(-svm_gamma * kernel);
    }
    return kernels;
}

int calculate_votes(float *kernels)
{
    float decision;
    int votes[NUM_CLASSES] = {0};
    int intercept_idx = 0;
    for (int m = 0; m < NUM_CLASSES - 1; m++)
    {
        for (int n = m + 1; n < NUM_CLASSES; n++)
        {
            decision = intercepts[intercept_idx];
            for (int p = w_sum[m]; p < w_sum[m + 1]; p++)
                decision += kernels[p] * coeffs[n - 1][p];

            for (int p = w_sum[n]; p < w_sum[n + 1]; p++)
                decision += kernels[p] * coeffs[m][p];
            votes[decision > 0 ? m : n] += 1;
            decision = 0;
            intercept_idx++;
        }
    }
    int val = votes[0];
    int idx = 0;

    for (int i = 1; i < NUM_CLASSES; i++)
    {
        if (votes[i] > val)
        {
            val = votes[i];
            idx = i;
        }
    }
    return idx;
}

#endif /* INC_SVM_INFERENCE_H_ */
