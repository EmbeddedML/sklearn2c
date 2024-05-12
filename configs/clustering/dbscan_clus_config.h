#ifndef DBSCAN_CLUS_CONFIG_H_INCLUDED
#define DBSCAN_CLUS_CONFIG_H_INCLUDED
#define NUM_CORE_POINTS 144
#define NUM_FEATURES 2
#define NUM_CLUSTERS 3
#define EPS 1
extern float CORE_POINTS[NUM_CORE_POINTS][NUM_FEATURES];
extern int LABELS[NUM_CORE_POINTS];
#endif