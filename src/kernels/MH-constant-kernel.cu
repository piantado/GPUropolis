/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 *
 * Run MCMC on constants only (not currently incorporating constant types)
 */

// A simplified version of MH_kernel that also does updates to the constant array
__global__ void MH_constant_kernel(int N, mcmc_specification* all_spec, mcmc_results* all_results)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose, impossible to debug
	
	mcmc_specification* spec   = &(all_spec[idx]);
	mcmc_results*       result = &(all_results[idx]); 
	
	// unpackage the specification
	int iterations               = spec->iterations;	
	float prior_temperature      = spec->prior_temperature;
	float likelihood_temperature = spec->likelihood_temperature;
	float acceptance_temperature = spec->acceptance_temperature;
	int data_length              = spec->data_length;
	datum* data                  = spec->data;
	int rx = spec->rng_seed;
	
	data_t stack[MAX_PROGRAM_LENGTH+1]; // Create the stack, allowing for one constant for each op to be passed (really we only need half this, right?)
	
	hypothesis current_, proposal_, current_MAP_;
	hypothesis* current     = &current_;
	hypothesis* proposal    = &proposal_;
	hypothesis* current_MAP = &current_MAP_;
	
	// copy to most local, current
	COPY_HYPOTHESIS(current, &(result->sample)); // copy to most local, current, to get all initialization check bits, etc
		
	// randomly initialize if we should
	if(spec->initialize){
		random_closed_expression(current,  RNG_ARGS); 
	}
	
	// update all the posterior, etc.
	update_hypothesis(current); // should go BEFORE compute_posterior
	compute_posterior(data_length, data, current, stack);
	
	// initialize everything to be the current (especially proposal and MAP)
	COPY_HYPOTHESIS( proposal, current);
	COPY_HYPOTHESIS( current_MAP, current);
	
	// Stats on MCMC
	int this_chain_acceptance_count = 0;
	int this_chain_proposals = 0;
	
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<iterations;mcmci++) {
			
		float fb = 0.0;  // the forward-backward probability
		
		COPY_CONSTANTS(proposal, current); // copy over
		
		// propose at random
		fb += resample_random_constant(proposal, RNG_ARGS);
		// no update hypothesis needed, because we don't change structure!
		
		// compute the posterior for the proposal
		compute_posterior(data_length, data, proposal, stack);
		this_chain_proposals += 1; // how many total proposals have we made?
		
		float pcur = current->prior/prior_temperature + current->likelihood/likelihood_temperature;
		float ppro = proposal->prior/prior_temperature + proposal->likelihood/likelihood_temperature;
		int swap = (is_valid(proposal->posterior) && random_float(RNG_ARGS) < exp( (ppro-pcur+fb)/acceptance_temperature ))
			   || is_invalid(current->posterior);
			   
		// swap if we should
		if(swap) {
			hypothesis* tmp;
			tmp = current;
			current  = proposal;
			proposal = tmp;
			
			// update the chain acceptance count
			this_chain_acceptance_count += 1;
			
			if(current->posterior > result->MAP.posterior || is_invalid(result->MAP.posterior)){
				COPY_HYPOTHESIS( &(result->MAP), current);
			}
		} // end if swap
	}
	
	COPY_HYPOTHESIS(&(result->sample), current);
	
}
 