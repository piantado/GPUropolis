/*


*/

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// These are used below but defined for real elsewhere
 float f_output(float x, hypothesis* h, float* stack);
 int find_program_length(int* program);
 int find_program_close(int* program);
 int find_close_backwards(int* buf, int pos);


 float compute_likelihood(int DLEN, float* device_INPUT, float* device_OUTPUT, float* device_SD, hypothesis* h, float* stack) {

	float ll = 0.0;
	for(int i=0;i<DLEN;i++){
		//__syncthreads(); // NOT a speed improvement
		
		// compute the difference between the output and what we see
		float d = device_OUTPUT[i] - f_output( device_INPUT[i], h, stack);
		
		// and use a gaussian likelihood
		ll += lnormalpdf(d, device_SD[i]);
	}
	
 	return ll / LL_TEMPERATURE;
}


 float compute_prior(hypothesis* h) {
	
// 	float lprior = compute_generation_probability(h); // the prior just on the program, and ad the rest
// 	return lprior / PRIOR_TEMPERATURE;
	
	// We just use the proposal as a prior; else compute it as above
	return  h->proposal_generation_lp / PRIOR_TEMPERATURE;
}

 void compute_posterior(int DLEN, float* device_INPUT, float* device_OUTPUT, float* device_SD, hypothesis* h, float* stack) {
	h->prior      = compute_prior(h);
	h->likelihood = compute_likelihood(DLEN, device_INPUT, device_OUTPUT, device_SD, h, stack);
	h->posterior = h->prior + h->likelihood;
}
