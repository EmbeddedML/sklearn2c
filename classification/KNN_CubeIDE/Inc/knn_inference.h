/*
 * knn_inference.h
 *
 *  Created on: Jan 22, 2022
 *      Author: berkan
 */

#ifndef INC_KNN_INFERENCE_H_
#define INC_KNN_INFERENCE_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "knn_config.h"

struct indexedArr
{
    float value;
    int index;
};

float euclid_distance(float sample[], float target[]);
int compare(const void *a, const void *b);
int knn_inference(float input[]);

#endif /* INC_KNN_INFERENCE_H_ */
