#include "bayes_inference.h"

int largest(float arr[]);
float *run_bayes_classifier(Matrix *input);

int largest(float arr[])
{
    float max = arr[0];
    int max_idx = 0;

    for (int i = 1; i < NUM_CLASSES; i++)
        if (arr[i] > max)
            max_idx = i;

    return max_idx;
}

float *run_bayes_classifier(Matrix *input)
{
    // Returns probabilities of the classes using Bayes Classifier
    Matrix *input_T, *mu, *sigma;
    input_T = transpose(input);
    static float probs[NUM_CLASSES] = {0};
    for (int cls = 0; cls < NUM_CLASSES; cls++)
    {
        mu = copy_matrix(MEANS[cls], NUM_FEATURES, 1);
        sigma = copy_matrix(INV_COVS[cls], NUM_FEATURES, NUM_FEATURES);
        Matrix *zero_mean = subtract(input, mu);
        Matrix *res1 = multiply(transpose(zero_mean), sigma);
        res1 = multiply(res1, zero_mean);
        probs[cls] = logf(CLASS_PRIORS[cls]) - logf(DETS[cls]) / 2 - res1->data[0][0] / 2;
    }

    free(sigma);
    free(mu);
    free(input_T);
    return probs;
}
