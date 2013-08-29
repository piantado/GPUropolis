/*
	Simple Bayes functions on hypotheses
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
	
 	return ll / LL_TEMPERATURE;
}


__device__ float compute_prior(hypothesis* h) {
	
// 	float lprior = compute_generation_probability(h); // the prior just on the program, and ad the rest
// 	return lprior / PRIOR_TEMPERATURE;
	
	// We just use the proposal as a prior
	// NOTE: This means compute_generation_probability MUST be called beforee compute_posterior
	return  h->proposal_generation_lp / PRIOR_TEMPERATURE;
}

__device__ void compute_posterior(int DLEN, datum* device_data, hypothesis* h, float* stack) {
	h->prior      = compute_prior(h);
	h->likelihood = compute_likelihood(DLEN, device_data, h, stack);
	h->posterior = h->prior + h->likelihood;
}
