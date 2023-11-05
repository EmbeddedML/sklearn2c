#include <stdio.h>
#include "dtc_inference.h"

int dtc_inference(float input[])
{
    int idx = 0; // Root Node
    while (idx >= 0)
    {
        int feature_val = input[SPLIT_FEATURE[idx]];
        if (SPLIT_FEATURE[idx] < 0)
            return VALUES[idx];
        if (feature_val < THRESHOLDS[idx])
            idx = LEFT_CHILDREN[idx];
        else
            idx = RIGHT_CHILDREN[idx];
    }
    return -1;
}
