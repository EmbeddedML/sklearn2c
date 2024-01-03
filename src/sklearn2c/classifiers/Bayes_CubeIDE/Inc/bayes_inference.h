/*
 * lib_bayes.h
 *
 *  Created on: Oct 6, 2023
 *     Authors: Berkan Höke, Eren Atmaca
 */

#ifndef INC_BAYES_INFERENCE_H_
#define INC_BAYES_INFERENCE_H_

#ifdef __cplusplus
extern "C"
#endif

#include "stm32f746xx.h"
#include "arm_math.h"
#include <bayes_conf.h>

int8_t BAYES_Classify(arm_matrix_instance_f32 * input, int case, arm_matrix_instance_f32 * output);


#ifdef __cplusplus
}
#endif


#endif /* INC_BAYES_INFERENCE_H_ */
