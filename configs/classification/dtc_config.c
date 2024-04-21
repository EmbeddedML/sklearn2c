#include "dtc_config.h"
const int LEFT_CHILDREN[NUM_NODES] = { 1, 2,-1, 4, 5, 6,-1, 8,-1,-1,-1,12,13,-1,-1,-1,17,18,-1,20,-1,-1,23,24,
 -1,-1,27,28,-1,-1,-1};
const int RIGHT_CHILDREN[NUM_NODES] = {16, 3,-1,11,10, 7,-1, 9,-1,-1,-1,15,14,-1,-1,-1,22,19,-1,21,-1,-1,26,25,
 -1,-1,30,29,-1,-1,-1} ;
const int SPLIT_FEATURE[NUM_NODES] = { 0, 1,-2, 1, 1, 0,-2, 0,-2,-2,-2, 1, 1,-2,-2,-2, 0, 1,-2, 0,-2,-2, 0, 0,
 -2,-2, 1, 1,-2,-2,-2};
const float THRESHOLDS[NUM_NODES] = {-0.40399808,-1.68660957,-2.        ,-1.15887409,-1.1878534 ,-0.57196411,
 -2.        ,-0.49871337,-2.        ,-2.        ,-2.        ,-1.02262264,
 -1.03035527,-2.        ,-2.        ,-2.        ,-0.0958238 ,-1.08932471,
 -2.        ,-0.17569117,-2.        ,-2.        , 0.04536033, 0.011741  ,
 -2.        ,-2.        ,-1.33956552,-1.34416306,-2.        ,-2.        ,
 -2.        };
const int VALUES[NUM_NODES][NUM_CLASSES] = {{0,0},
 {0,0},
 {0,1},
 {0,0},
 {0,0},
 {0,0},
 {1,0},
 {0,0},
 {0,1},
 {1,0},
 {0,1},
 {0,0},
 {0,0},
 {1,0},
 {0,1},
 {1,0},
 {0,0},
 {0,0},
 {1,0},
 {0,0},
 {0,1},
 {1,0},
 {0,0},
 {0,0},
 {0,1},
 {1,0},
 {0,0},
 {0,0},
 {0,1},
 {1,0},
 {0,1}};
