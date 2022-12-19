/*
 * reg_tree_config.h
 *
 *  Created on: Dec 8, 2022
 *      Author: hp
 */

#ifndef INC_REG_TREE_CONFIG_H_
#define INC_REG_TREE_CONFIG_H_


#define NUM_FEATURES 2
#define NUM_NODES 99

extern const int LEFT_CHILDREN[NUM_NODES];
extern const int RIGHT_CHILDREN[NUM_NODES];
extern const int SPLIT_FEATURE[NUM_NODES];
extern const float THRESHOLDS[NUM_NODES];
extern const float VALUES[NUM_NODES];

#endif /* INC_REG_TREE_CONFIG_H_ */
