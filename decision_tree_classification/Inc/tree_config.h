/*
 * tree_config.h
 *
 *  Created on: Jan 22, 2022
 *      Author: berkan
 */

#ifndef INC_TREE_CONFIG_H_
#define INC_TREE_CONFIG_H_

#define NUM_CLASSES 3
#define NUM_FEATURES 2
#define NUM_NODES 11
const char* LABELS[NUM_CLASSES] = {"CLASS 1", "CLASS 2", "CLASS 3"};
int LEFT_CHILDREN[NUM_NODES] = { 1,  2, -1,  4, -1, -1,  7, -1,  9, -1, -1};
int RIGHT_CHILDREN[NUM_NODES] = { 6,  3, -1,  5, -1, -1,  8, -1, 10, -1, -1};
int SPLIT_FEATURE[NUM_NODES] = { 0,  1, -2,  0, -2, -2,  1, -2,  1, -2, -2};
float THRESHOLDS[NUM_NODES] = { 4.03142989,  3.7743994 , -2.        ,  2.85007179, -2.        ,
 -2.        ,  2.24236071, -2.        ,  2.61266327, -2.        ,
 -2.        };
int VALUES[NUM_NODES] = {1, 0, 0, 2, 2, 0, 1, 1, 0, 0, 1};



#endif /* INC_TREE_CONFIG_H_ */
