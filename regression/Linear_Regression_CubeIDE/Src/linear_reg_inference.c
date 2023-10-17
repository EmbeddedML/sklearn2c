#include "linear_reg_inference.h"

float linear_reg_inference(float input[])
{
    float sum = OFFSET;
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        sum += COEFFS[i] * input[i];
    }

    return sum;
}