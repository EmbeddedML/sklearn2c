/*
 * reg_tree_config.c
 *
 *  Created on: Dec 8, 2022
 *      Author: hp
 */

#include "reg_tree_config.h"

int LEFT_CHILDREN[NUM_NODES] = { 1, 2, 3,-1, 5,-1, 7,-1,-1,10,11,12,-1,14,15,-1,-1,18,-1,-1,21,22,23,-1,
 -1,-1,27,28,-1,-1,-1,32,33,34,-1,36,-1,38,39,-1,-1,-1,43,44,-1,-1,-1,48,
 -1,50,51,-1,-1,54,55,-1,-1,-1,59,60,61,62,-1,-1,65,66,-1,68,-1,-1,-1,72,
 73,-1,75,-1,-1,78,-1,-1,81,82,83,84,85,-1,-1,-1,-1,90,-1,92,93,-1,-1,-1,
 97,-1,-1};
int RIGHT_CHILDREN[NUM_NODES] = {58, 9, 4,-1, 6,-1, 8,-1,-1,31,20,13,-1,17,16,-1,-1,19,-1,-1,26,25,24,-1,
 -1,-1,30,29,-1,-1,-1,47,42,35,-1,37,-1,41,40,-1,-1,-1,46,45,-1,-1,-1,49,
 -1,53,52,-1,-1,57,56,-1,-1,-1,80,71,64,63,-1,-1,70,67,-1,69,-1,-1,-1,77,
 74,-1,76,-1,-1,79,-1,-1,96,89,88,87,86,-1,-1,-1,-1,91,-1,95,94,-1,-1,-1,
 98,-1,-1} ;
int SPLIT_FEATURE[NUM_NODES] = { 1, 1, 1,-2, 0,-2, 0,-2,-2, 1, 1, 1,-2, 0, 1,-2,-2, 0,-2,-2, 1, 1, 1,-2,
 -2,-2, 1, 0,-2,-2,-2, 1, 1, 1,-2, 1,-2, 0, 1,-2,-2,-2, 0, 0,-2,-2,-2, 1,
 -2, 1, 0,-2,-2, 1, 0,-2,-2,-2, 1, 1, 1, 1,-2,-2, 1, 1,-2, 0,-2,-2,-2, 1,
  1,-2, 0,-2,-2, 1,-2,-2, 1, 1, 1, 1, 1,-2,-2,-2,-2, 1,-2, 0, 1,-2,-2,-2,
  1,-2,-2};
float THRESHOLDS[NUM_NODES] = { 1.36976224e-01,-1.30643535e+00,-2.30110538e+00,-2.00000000e+00,
 -7.55604208e-02,-2.00000000e+00, 3.85074958e-01,-2.00000000e+00,
 -2.00000000e+00,-6.17545635e-01,-9.35861886e-01,-1.09305853e+00,
 -2.00000000e+00,-2.33424507e-01,-1.04353660e+00,-2.00000000e+00,
 -2.00000000e+00,-1.54387347e-01,-2.00000000e+00,-2.00000000e+00,
 -7.98853755e-01,-8.23265523e-01,-8.47411692e-01,-2.00000000e+00,
 -2.00000000e+00,-2.00000000e+00,-7.16252983e-01,-1.85093284e-03,
 -2.00000000e+00,-2.00000000e+00,-2.00000000e+00,-3.08246806e-01,
 -4.69012156e-01,-5.39270639e-01,-2.00000000e+00,-4.99063686e-01,
 -2.00000000e+00,-4.16904688e-04,-4.95985791e-01,-2.00000000e+00,
 -2.00000000e+00,-2.00000000e+00, 5.20329576e-01,-4.06765807e-01,
 -2.00000000e+00,-2.00000000e+00,-2.00000000e+00,-2.14614399e-01,
 -2.00000000e+00,-1.01798661e-01,-1.66239887e-01,-2.00000000e+00,
 -2.00000000e+00,-5.23346132e-02, 6.18278086e-01,-2.00000000e+00,
 -2.00000000e+00,-2.00000000e+00, 8.59400541e-01, 5.93176067e-01,
  3.91153872e-01, 3.40780556e-01,-2.00000000e+00,-2.00000000e+00,
  5.38666576e-01, 4.56898168e-01,-2.00000000e+00,-1.81608256e-01,
 -2.00000000e+00,-2.00000000e+00,-2.00000000e+00, 7.50517279e-01,
  6.54959589e-01,-2.00000000e+00, 1.56414449e-01,-2.00000000e+00,
 -2.00000000e+00, 8.14550012e-01,-2.00000000e+00,-2.00000000e+00,
  1.26318586e+00, 9.95328605e-01, 9.47822690e-01, 9.21465039e-01,
  9.00833070e-01,-2.00000000e+00,-2.00000000e+00,-2.00000000e+00,
 -2.00000000e+00, 1.04253757e+00,-2.00000000e+00, 1.13981384e+00,
  1.11892807e+00,-2.00000000e+00,-2.00000000e+00,-2.00000000e+00,
  1.60936314e+00,-2.00000000e+00,-2.00000000e+00};
float VALUES[NUM_NODES] = {  93.01288255,  43.13637549, -46.97929025,-107.95428123, -26.65429325,
  -13.81278655, -33.0750466 , -32.34754578, -33.80254743,  57.55488201,
   33.71489331,  23.24042766,  26.33354146,  22.4671492 ,  23.60002436,
   22.85727698,  24.34277173,  21.33427405,  20.50559995,  22.16294816,
   42.44361469,  38.25216909,  36.51954163,  36.00185644,  37.03722683,
   41.717424  ,  46.6350603 ,  44.12633288,  42.28583467,  45.96683108,
   51.65251514,  76.2863017 ,  64.90576897,  62.41460918,  57.19561821,
   63.71935692,  62.07666229,  64.26692179,  63.93645488,  63.81640884,
   64.05650093,  64.9278556 ,  69.05770197,  69.55330762,  69.19359159,
   69.91302364,  68.06649068,  91.46034533,  80.15195185,  93.72202402,
   90.08210749,  90.7369209 ,  89.42729408,  96.14863504,  94.92931277,
   94.7340647 ,  95.12456085,  98.58727959, 161.88996372, 142.68149124,
  133.6752177 , 125.62382946, 124.44981094, 126.79784798, 137.70091181,
  136.20413263, 134.11044944, 137.25097422, 136.68896704, 137.81298141,
  142.19124938, 153.4890195 , 150.43140821, 146.6937222 , 152.30025121,
  153.10918793, 151.4913145 , 158.07543644, 158.08284876, 158.06802413,
  183.01928344, 173.99811951, 167.19593909, 166.04169493, 165.29211568,
  165.21069427, 165.37353709, 167.54085342, 170.65867157, 180.80029993,
  175.17672089, 182.67482628, 184.02440881, 184.30828377, 183.74053385,
  179.97566122, 219.10393917, 201.22967472, 236.97820362};
