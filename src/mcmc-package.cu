/* Passed back and forth to CUDA, giving everything needed to run */

typedef struct mcmc_package {
	
	hypothesis sample;
	hypothesis MAP;
	hypothesis proposal; // and a memory slot for proposal
	
	int rng_seed; 
	int chain_index;
	
	// and pointers to the data we want
	int data_length;
	datum* data;
	
	int acceptance_count;
	int proposal_count;
	
} mcmc_package;