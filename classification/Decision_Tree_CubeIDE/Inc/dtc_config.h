#ifndef DTC_CONFIG_H_INCLUDED
#define DTC_CONFIG_H_INCLUDED
#define NUM_NODES 51
#define NUM_FEATURES 2
extern const int LEFT_CHILDREN[NUM_NODES];
extern const int RIGHT_CHILDREN[NUM_NODES];
extern const int SPLIT_FEATURE[NUM_CLASSES];
extern const float THRESHOLDS[NUM_CLASSES];
extern const float VALUES[NUM_NODES];
#endif
