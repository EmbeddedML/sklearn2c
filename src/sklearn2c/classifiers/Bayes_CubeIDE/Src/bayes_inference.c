/*
 * lib_bayes.c
 *
 *  Created on: Oct 6, 2023
 *     Authors: Berkan Höke, Eren Atmaca
 */

#include <bayes_inference.h>

int8_t BAYES_Classify(arm_matrix_instance_f32 *input, int case, arm_matrix_instance_f32 *output)
{
	// Returns probabilities of the classes using Bayes Classifier

	int8_t status = ARM_MATH_SUCCESS;

	float32_t __input_T[NUM_FEATURES * 1];
	float32_t __mu_T[NUM_FEATURES * 1];
	float32_t __zero_mean[NUM_FEATURES * 1];
	float32_t __zero_mean_T[NUM_FEATURES * 1];
	float32_t __xt_sigma[NUM_FEATURES * 1];

	float32_t __xt_sigma_x;
	float32_t __sigma_mu[NUM_FEATURES * 1];
	float32_t __sigma_mu_T[NUM_FEATURES * 1];
	float32_t __sigma_mu_x;

	float32_t __mu_sigma_mu;

	arm_matrix_instance_f32 mu, sigma;
	arm_matrix_instance_f32 mu_T = {.numRows = 1, .numCols = NUM_FEATURES, .pData = __mu_T};

	arm_matrix_instance_f32 input_T = {.numRows = 1, .numCols = NUM_FEATURES, .pData = __input_T};
	arm_matrix_instance_f32 zero_mean = {.numRows = NUM_FEATURES, .numCols = 1, .pData = __zero_mean};
	arm_matrix_instance_f32 zero_mean_T = {.numRows = 1, .numCols = NUM_FEATURES, .pData = __zero_mean_T};
	arm_matrix_instance_f32 xt_sigma = {.numRows = 1, .numCols = NUM_FEATURES, .pData = __xt_sigma};
	arm_matrix_instance_f32 xt_sigma_x = {.numRows = 1, .numCols = 1, .pData = &__xt_sigma_x};
	arm_matrix_instance_f32 sigma_mu = {.numRows = NUM_FEATURES, .numCols = 1, .pData = __sigma_mu};
	arm_matrix_instance_f32 sigma_mu_T = {.numRows = 1, .numCols = NUM_FEATURES, .pData = __sigma_mu_T};
	arm_matrix_instance_f32 sigma_mu_x = {.numRows = 1, .numCols = 1, .pData = &__sigma_mu_x};
	arm_matrix_instance_f32 mu_sigma_mu = {.numRows = 1, .numCols = 1, .pData = &__mu_sigma_mu};

	status = arm_mat_trans_f32(input, &input_T);

	float discr[NUM_CLASSES] = {0};
	for (int cls = 0; cls < NUM_CLASSES; cls++)
	{
		mu.pData = &MEANS[cls][0];
		mu.numRows = NUM_FEATURES;
		mu.numCols = 1;
		float prior = logf(CLASS_PRIORS[cls]);

		sigma.pData = &INV_COVS[cls][0];
		sigma.numRows = NUM_FEATURES;
		sigma.numCols = NUM_FEATURES;
		status += arm_mat_sub_f32(&input, &mu, &zero_mean);
		status += arm_mat_trans_f32(zero_mean, &zero_mean_T);

		if (case == 1)
		{
			status += arm_mat_mult_f32(&zero_mean, &zero_mean_T, &prod);
			discr[cls] = prior - prod.pData[0] / (2 * sigma_sq);
		}
		else if (case == 2)
		{
			status += arm_mat_mult_f32(zero_mean, inv_cov, &prod);
			status += arm_mat_mult_f32(prod, zero_mean_T, prod);
			discr[cls] = prior - prod * 0.5;
		}
		else if (case == 3)
		{
			status += arm_mat_mult_f32(&input_T, &sigma, &xt_sigma);
			status += arm_mat_mult_f32(&xt_sigma, input, &xt_sigma_x);

			xt_sigma_x.pData[0] = xt_sigma_x.pData[0] * (-0.5f);

			status += arm_mat_mult_f32(&sigma, &mu, &sigma_mu);
			status += arm_mat_trans_f32(&sigma_mu, &sigma_mu_T);
			status += arm_mat_mult_f32(&sigma_mu_T, input, &sigma_mu_x);
			status += arm_mat_trans_f32(&mu, &mu_T);
			status += arm_mat_mult_f32(&mu_T, &sigma_mu, &mu_sigma_mu);

			// mu_sigma_mu.pData[0] = mu_sigma_mu.pData[0] * (-0.5f);

			float log_det = logf(DETS[cls]) * (-0.5f);

			discr[cls] = xt_sigma_x.pData[0] + sigma_mu_x.pData[0] - 0.5f * mu_sigma_mu.pData[0] + log_det + prior;
		}

		else
		{
			return -1;
		}
	}
	memcpy(output->pData, discr, sizeof(float32_t) * NUM_CLASSES);
	output->numCols = NUM_CLASSES;
	output->numRows = 1;
	return status;
}
