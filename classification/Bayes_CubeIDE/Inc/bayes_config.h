#ifndef BAYES_CONFIG_H_INCLUDED
#define BAYES_CONFIG_H_INCLUDED
#define NUM_CLASSES 2
#define NUM_FEATURES 2
extern const char *LABELS[NUM_CLASSES];
extern float MEANS[NUM_CLASSES][NUM_FEATURES];
extern float INV_COVS[NUM_CLASSES][NUM_FEATURES * NUM_FEATURES];
extern const float DETS[NUM_CLASSES];
extern const float CLASS_PRIORS[NUM_CLASSES];
#endif
