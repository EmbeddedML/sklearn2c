#include <stdbool.h>
#include <float.h>
#include "dbscan_inference.h"

float euclid_distance(float sample[], float target[])
{
    float dist = 0;
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        float diff = sample[i] - target[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

int dbscan_predict(float input[])
{
    int cur_label_idx = 0;
    int label = -1;
    float min_distance = FLT_MAX;
    for (int i = 0; i < NUM_CORE_POINTS; i++)
    {
        float cur_dist = euclid_distance(input, CORE_POINTS[i]);
        if (cur_dist < min_distance)
        {
            min_distance = cur_dist;
            cur_label_idx = i;
        }
    }
    
    if (min_distance < EPS)
        label = LABELS[cur_label_idx];
        
    return label;
}