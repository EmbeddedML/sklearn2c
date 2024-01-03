#include "bayes_config.h"
const char *LABELS[NUM_CLASSES] = {"CLASS 1","CLASS 2"};
float MEANS[NUM_CLASSES][NUM_FEATURES] = {{2.40542682,2.04139237},
 {0.95042967,1.99338236}};
float INV_COVS[NUM_CLASSES][NUM_FEATURES * NUM_FEATURES] = {{ 1.40938779,-0.04605597,-0.04605597, 0.26668508},
 { 5.59177271,-0.11959348,-0.11959348, 0.98673676}};
const float DETS[NUM_CLASSES] = {0.61134405,2.34591242};
const float CLASS_PRIORS[NUM_CLASSES] = {0.49375,0.50625};