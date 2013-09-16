/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Simple bayes functions on hypotheses
 */

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// These are used below but defined for real elsewhere
__device__ float f_output(float x, hypothesis* h, float* stack);
__device__ int find_program_length(int* program);
__device__ int find_program_close(int* program);
__device__ int find_close_backwards(int* buf, int pos);


__device__ float compute_likelihood(int DLEN, datum* device_data, hypothesis* h, float* stack) {

	float ll = 0.0;
	for(int i=0;i<DLEN;i++){
		//__syncthreads(); // NOT a speed improvement
		
		// compute the difference between the output and what we see
		data_t d = device_data[i].output - f_output( device_data[i].input, h, stack);
		
		// and use a gaussian likelihood
		ll += lnormalpdf(d, device_data[i].sd);
	}
	
 	h->likelihood = ll / LL_TEMPERATURE;
	
	return h->likelihood;
	
}


__device__ float compute_prior(hypothesis* h) {
	
// 	float lprior = compute_generation_probability(h); // the prior just on the program, and ad the rest
// 	return lprior / PRIOR_TEMPERATURE;
	
	// We just use the proposal as a prior
	// NOTE: This means compute_generation_probability MUST be called beforee compute_posterior
 	h->prior =  h->proposal_generation_lp / PRIOR_TEMPERATURE;
	
	// The fancy other prior
// 	h->prior = compute_x1depth_prior(h) / PRIOR_TEMPERATURE;
	
	return h->prior;
}

__device__ void compute_posterior(int DLEN, datum* device_data, hypothesis* h, float* stack) {
	compute_prior(h);
	compute_likelihood(DLEN, device_data, h, stack);
	h->posterior = h->prior + h->likelihood;
}
