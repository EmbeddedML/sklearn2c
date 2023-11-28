#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "svm_config.h"
#include "svm_inference.h"

float kernels[NUM_SV] = {0};


float *compute_kernels(float *x)
{
    memset(kernels, 0, sizeof(kernels));
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

float *calculate_ovr_scores(float *confidences){
        int votes[NUM_CLASSES] = {0};
        sum_of_conf[NUM_CLASSES] = {0};

        int k = 0
        for i in range(NUM_CLASSES):
            for j in range(i + 1, NUM_CLASSES):
                sum_of_conf[i] += confidences[k]
                sum_of_confidences[:, j] -= confidences[k]
                votes[predictions[:, k] == 0, i] += 1
                votes[predictions[:, k] == 1, j] += 1
                k += 1

        transformed_confidences = sum_of_confidences / (3 * (abs(sum_of_confidences) + 1));
        return votes + transformed_confidences
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
