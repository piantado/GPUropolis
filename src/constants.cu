// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Manipulate constants in hypotheses
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Take h and resample one of its constants according to the types
// and return the f/b probability
// __device__ float resample_random_constant(hypothesis* h, RNG_DEF) {
// 	
// 	int k = random_int(MAX_CONSTANTS, RNG_ARGS);
// 	
// 	float old_value = h->constants[k];
// 	float new_value = 0.0;
// 	float fb = 0.0;
// 	switch(h->constant_types[k]) {
// 		case GAUSSIAN:
// 			new_value = random_normal(RNG_ARGS);
// 			fb = lnormalpdf(new_value, 1.0) - lnormalpdf(old_value, 1.0); // these and all constants are set to the generation defaults
// 			break;
// 		case LOGNORMAL:
// 			new_value = random_lnormal(0.0, 1.0, RNG_ARGS);
// 			fb = llnormalpdf(new_value, 1.0) - llnormalpdf(old_value, 1.0);
// 			break;
// 		case EXPONENTIAL:
// 			new_value = random_exponential(1.0, RNG_ARGS);
// 			fb = lexponentialpdf(new_value, 1.0) - lexponentialpdf(old_value, 1.0);
// 			break;
// 		case UNIFORM:
// 			new_value = random_float(RNG_ARGS);
// 			fb = luniformpdf(new_value) - luniformpdf(old_value);
// 			break;			
// 	}
// 	h->constants[k] = new_value;
// 	return fb;	
// }


// a drift kernel
__device__ float resample_random_constant(hypothesis* h, RNG_DEF) {
	if(h->nconstants == 0) return 0.0;
	
	int k = random_int(h->nconstants, RNG_ARGS);
	
	float old_value = h->constants[k];
	float new_value = old_value + 0.1*random_normal(RNG_ARGS); // just drive a standard gaussian
	h->constants[k] = new_value;
	
	// since it's symmetric drift kernel proposal,
	return 0.0;	
}

// Take h and resample one of the constant *types*
// returning fb
// TODO: WE CAN MAKE THIS A local-GIBBS MOVE--compute the probability of the observed constant
// under each type and resample!!
__device__ float resample_random_constant_type(hypothesis* h, RNG_DEF) {
	if(h->nconstants == 0) return 0.0;
	
	int k = random_int(h->nconstants, RNG_ARGS);
	h->constant_types[k] = random_int(__N_CONSTANT_TYPES, RNG_ARGS);
	return 0.0;
}

// Randomize the constants
__device__ float randomize_constants(hypothesis* h, RNG_DEF) {
	for(int k=0;k<MAX_CONSTANTS;k++) {
		h->constants[k] = random_normal(RNG_ARGS);
		h->constant_types[k] = random_int(__N_CONSTANT_TYPES, RNG_ARGS);
	}
	return 0.0;
}

// Compute the prior on constats
// assuming constant types are generated uniformly
__device__ float compute_constants_prior(hypothesis* h) {
	
	float lp = 0.0;
	for(int k=0;k<h->nconstants;k++) {
		float value = h->constants[k];
		
		switch(h->constant_types[k]) {
			case GAUSSIAN:    lp += lnormalpdf(value, 1.0); break;
// 			case LOGNORMAL:   lp += llnormalpdf(value, 1.0); break;
			case EXPONENTIAL: lp += lexponentialpdf(value, 1.0); break;
			case UNIFORM:     lp += luniformpdf(value); break;	
		}
	}
	
	return lp;
}
