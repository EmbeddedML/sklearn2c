#include "bayes_config.h"

const char *LABELS[NUM_CLASSES] = {"CLASS 1", "CLASS 2", "CLASS 3"};
float MEANS[NUM_CLASSES][NUM_FEATURES] = {{2.87674415, 2.25784206},
                                          {5.060645, 0.9774933},
                                          {0.93544251, 4.94022031}};
float INV_COVS[NUM_CLASSES][NUM_FEATURES * NUM_FEATURES] = {{1.32880985, -0.00671077, -0.00671077, 0.23575473},
                                                            {3.38970586, 0.18949161, 0.18949161, 1.47002762},
                                                            {1.06499168, -0.10310861, -0.10310861, 4.31166234}};
const float DETS[NUM_CLASSES] = {0.55966792, 2.22419742, 2.14038621};
float CLASS_PRIORS[NUM_CLASSES] = {0.325, 0.35, 0.325};
