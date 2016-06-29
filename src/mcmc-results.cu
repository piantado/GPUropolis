/*

	Return values for an mcmc run
*/

typedef struct mcmc_results {
	
	hypothesis sample;
	hypothesis MAP;
	
	// for storing mcmc acceptance ratios, etc. 
	unsigned int acceptance_count;
	unsigned int proposal_count;
	
	int rng_seed; // what *was* I run with? Given by spec
	
} mcmc_results;