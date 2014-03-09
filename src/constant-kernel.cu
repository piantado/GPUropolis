/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 *
 * Run MCMC on constants only (not currently incorporating constant types)
 */

// A simplified version of MH_kernel that also does updates to the constant array
__global__ void MH_constant_kernel(int N, mcmc_package* all_package, int iterations)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose, impossible to debug
	
	mcmc_package* mypackage   = &(all_package[idx]);
	
	// unpackage the specification
	hypothesis* sample           = &(mypackage->sample);
	hypothesis* proposal         = &(mypackage->proposal);
	hypothesis* MAP              = &(mypackage->MAP);
	int data_length              = mypackage->data_length;
	datum* data                  = mypackage->data;
	int rx                       = mypackage->rng_seed;
	
	data_t stack[MAX_PROGRAM_LENGTH+1]; // Create the stack, allowing for one constant for each op to be passed (really we only need half this, right?)
	
	// update the sample
	update_posterior(data_length, data, sample, stack);
	
	// Stats on MCMC
	int this_chain_acceptance_count = 0;
	int this_chain_proposals = 0;
	
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<iterations;mcmci++) {
			
		float fb = 0.0;  // the forward-backward probability
		
		COPY_CONSTANTS(proposal, sample); // copy over
		
		// propose at random
		fb += resample_random_constant(proposal, RNG_ARGS);
		// no update hypothesis needed, because we don't change structure!
		
		// compute the posterior for the proposal
		update_posterior(data_length, data, proposal, stack);
		
		this_chain_proposals += 1; // how many total proposals have we made?
		
		int swap = is_valid(proposal->posterior) && random_float(RNG_ARGS) < exp( (proposal->posterior-sample->posterior+fb));
		
		// swap if we should
		if(swap || is_invalid(sample->posterior) ) {
			hypothesis* tmp = sample;
			sample  = proposal;
			proposal = tmp;
			
			// update the chain acceptance count
			this_chain_acceptance_count += 1;
			
			if(sample->posterior > MAP->posterior || is_invalid(MAP->posterior)){
				COPY_HYPOTHESIS(MAP, sample);
			}
		} // end if swap
	}
	
	// if we get to the end and we are pointing at mypackage->proposal, we need to 
	// swap them to make sure mypackage holds the right thing
	if(sample != &(mypackage->sample) ) {
		COPY_HYPOTHESIS( &(mypackage->sample), sample);	
	}
	
	// and update these counts
	mypackage->acceptance_count += this_chain_acceptance_count;
	mypackage->proposal_count += iterations;
}
 