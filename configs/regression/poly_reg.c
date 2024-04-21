#include "poly_reg.h"
const float COEFFS[NUM_FEATURES] = { 2.76287985e+01, 4.87922575e+00, 1.90930267e-15, 8.75401459e-15,
  2.96191703e-15,-8.09447016e-16, 2.11888531e-14, 1.69410452e-15,
 -1.20805690e-16};
const float OFFSET = 6.661338147750939e-16;
char *feature_names[NUM_FEATURES] = {"0","1","0*0","1","1*1","0*0*0","1","1*1","1*1*1"};
