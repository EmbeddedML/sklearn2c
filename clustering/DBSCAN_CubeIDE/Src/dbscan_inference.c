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
    return dist;
}

int max_index(int arr[])
{
    int max_idx = -1;
    int cur_max = 0;
    for (int idx = 0; idx < NUM_CLUSTERS; ++idx)
    {
        if (arr[idx] > cur_max)
        {
            cur_max = arr[idx];
            max_idx = idx;
        }
    }
    return max_idx;
}

int run_dbscan(float input[])
{
    int cluster_idx;
    int cluster_votes[NUM_CLUSTERS] = {0};
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        float cur_dist = euclid_distance(input, samples[i]);
        if (cur_dist < EPS)
        {
            cluster_votes[labels[i]]++;
        }
    }

    cluster_idx = max_index(cluster_votes);
    return cluster_idx;
}
