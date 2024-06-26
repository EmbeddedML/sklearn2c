#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "knn_inference.h"
#include <string.h>

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

struct indexedArr
{
    float value;
    int index;
};

int compare(const void *a, const void *b)
{
    struct indexedArr *a1 = (struct indexedArr *)a;
    struct indexedArr *a2 = (struct indexedArr *)b;
    if ((*a1).value < (*a2).value)
        return -1;
    else if ((*a1).value > (*a2).value)
        return 1;
    else
        return 0;
}

int knn_inference(float *input, int *output)
{
    int k = 0, max = 0;
    struct indexedArr dists[NUM_SAMPLES];
    for (int i = 0; i < NUM_SAMPLES; i++)
    {
        dists[i].value = euclid_distance(DATA[i], input);
        dists[i].index = i;
    }

    qsort(dists, NUM_SAMPLES, sizeof(struct indexedArr), compare);

    int votes[NUM_CLASSES] = {0};
    for (int i = 0; i < NUM_NEIGHBORS; i++)
        votes[DATA_LABELS[dists[i].index]]++;

    memcpy(output, votes, NUM_CLASSES * sizeof(int));
    return 0;

//
//    for (int i = 0; i < NUM_CLASSES; ++i)
//    {
//        if (votes[i] > max)
//        {
//            max = votes[i];
//            k = i;
//        }
//    }
//    return k;

}
