/*

	Specifications for an mcmc run. This way we can hand off different ones to different chains easily

*/

typedef struct mcmc_specification {
	float prior_temperature;
	float likelihood_temperature;
	float acceptance_temperature;
	unsigned int iterations; // how many steps to run?
	int initialize; // shoudl we re-initialize?	
	int rng_seed;
	
	// and pointers to the data we want
	int data_length;
	datum* data;
	
} mcmc_specification;