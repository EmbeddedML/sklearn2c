#ifndef SVC_CONFIG_H_INCLUDED
#define SVC_CONFIG_H_INCLUDED
#define NUM_CLASSES 4
#define NUM_INTERCEPTS 6
#define NUM_FEATURES 2
#define NUM_SV 135
extern const char *LABELS[NUM_CLASSES];
extern const float coeffs[NUM_CLASSES - 1][NUM_SV];
extern const float SV[NUM_SV][NUM_FEATURES];
extern const float intercepts[NUM_INTERCEPTS];
extern const float w_sum[NUM_CLASSES + 1];
extern const float svm_gamma;
#endif