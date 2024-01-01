#include <math.h>
#include "svc_config.h"
#include "svc_inference.h"

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

int calculate_ovo_scores(float *kernels, float *ovo_confs)
{
    int idx = 0;
    ovo_confs = memcpy(ovo_confs, intercepts, sizeof(float) * NUM_INTERCEPTS);
    for (int m = 0; m < NUM_CLASSES - 1; m++)
    {
        for (int n = m + 1; n < NUM_CLASSES; n++)
        {
            for (int p = w_sum[m]; p < w_sum[m + 1]; p++)
                ovo_confs[idx] += kernels[p] * coeffs[n - 1][p];

            for (int p = w_sum[n]; p < w_sum[n + 1]; p++)
                ovo_confs[idx] += kernels[p] * coeffs[m][p];
            idx++;
        }
    }

    return 0;
}

int calculate_ovr_scores(float ovo_confs[NUM_INTERCEPTS], float ovr_confs[NUM_CLASSES]){
    int votes[NUM_CLASSES] = {0};
    float sum_of_confs[NUM_CLASSES] = {0};

    int k = 0;
    for (int i = 0; i < NUM_CLASSES; i++){
        for(int j= i+1; j < NUM_CLASSES; j++){
            sum_of_confs[i] += ovo_confs[k];
            sum_of_confs[j] -= ovo_confs[k];
            votes[(ovo_confs[k] > 0) ? i : j] += 1;
            k += 1;
        }
    }
    
    for (int i = 0; i < NUM_CLASSES; i++)
        ovr_confs[i] = votes[i] + sum_of_confs[i] / (3 * (fabs(sum_of_confs[i]) + 1));
}

int findMax(float array[], int *max_idx) {
    float max_val = array[0];

    for (int i = 1; i < NUM_CLASSES; ++i) {
        if (array[i] > max_val) {
            *max_idx = i;
        }
    }
    
    return 0;
}


int svc_predict(float* input, char* output_label){
    float *kernels = compute_kernels(input);
    float ovo_scores[NUM_INTERCEPTS];
    float ovr_scores[NUM_CLASSES];
    int max_idx;
    int ovo_err = calculate_ovo_scores(kernels, ovo_scores);
    int ovr_err = calculate_ovr_scores(ovo_scores, ovr_scores);
    findMax(ovr_scores, &max_idx);
    strcpy(output_label, LABELS[max_idx]);
    return 0;
}
