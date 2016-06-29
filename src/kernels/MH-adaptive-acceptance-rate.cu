/*
 * GPUropolis - 2013 Dec 15 - Steve Piantadosi 
 *
 * This adaptively changes the temperature to achieve a given acceptance rate.
 * It stores HISTORY_SIZE previous log p of acceptance and adjusts; the problem is that it seems to spiral
 * out of control towards high temps, since high temps have crazier proposals, leading to a need for higher temperature, etc.
 * 
 * 
 */

// MH kernel with just RR regeneration proposal (for simplicity)

// right choice of these can make you accept any proposal after some time in a rut...
__device__ const float TARGET_ACCEPTANCE = 1e-9;
__device__ const int HISTORY_SIZE = 10; // how many do we keep?



__global__ void MH_adaptive_acceptance_rate(int N, mcmc_specification* all_spec, mcmc_results* all_results)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose
	
	// this stores the previous HISTORY_SIZE sampling values
	float history[HISTORY_SIZE];
	float history_sum = 0.0;
	int history_z = 0; // the normalizer for how much history we've filled in
	int history_idx = 0;
	
	
	mcmc_specification* spec   = &(all_spec[idx]);
	mcmc_results*       result = &(all_results[idx]); 
	
	// unpackage the specification
	int iterations               = spec->iterations;	
	float prior_temperature      = spec->prior_temperature;
	float likelihood_temperature = spec->likelihood_temperature;
	float acceptance_temperature = spec->acceptance_temperature;
	int data_length              = spec->data_length;
	datum* data                  = spec->data;
	
	// set up our data stack array
	data_t stack[MAX_PROGRAM_LENGTH+1];
	
	// Set up for the RNG
	int rx = spec->rng_seed;
	
	hypothesis current_, proposal_, tmpbuf1_;
	hypothesis *current=&current_, *proposal=&proposal_, *tmpbuf1=&tmpbuf1_;
	
	COPY_HYPOTHESIS(current, &(result->sample)); // copy to most local, current, to get all initialization check bits, etc
	
	// Initialize at random, or copy back from result (to start a previous chain)
	if(spec->initialize){
		random_closed_expression(current,  RNG_ARGS); 
		update_hypothesis(current); // should go BEFORE compute_posterior
	}
	
	current->chain_index = idx;	
	compute_posterior(data_length, data, current, stack);
	
	// initialize everything to be the current
	COPY_HYPOTHESIS( proposal,          current );
	COPY_HYPOTHESIS( &(result->sample), current );
	COPY_HYPOTHESIS( &(result->MAP),    current );
	
	// Stats on MCMC
	result->acceptance_count = 0;
	result->proposal_count = 0;
	result->rng_seed = spec->rng_seed; // store this
	
	// ---------------------------------------------------------------------------------
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<iterations;mcmci++) {
		
		float fb = 0.0;
		
		// generate a novel tree in tmpbuf1
		random_closed_expression(tmpbuf1, RNG_ARGS);
		
		// replace a subtree in proposal with tmpbuf1
		replace_random_subnode_with(current, proposal, tmpbuf1, RNG_ARGS); 
		
		// update the proposal's details
		update_hypothesis(proposal);
		
		// forward-back probability
		fb += (nlogf(current->program_length)+proposal->proposal_generation_lp );
		fb -= (nlogf(proposal->program_length)+current->proposal_generation_lp );
		
		// compute the posterior (setting prior, likelihood on proposal)
		compute_posterior(data_length, data, proposal, stack);
		
		float pcur = current->prior/prior_temperature + current->likelihood/likelihood_temperature;
		float ppro = proposal->prior/prior_temperature + proposal->likelihood/likelihood_temperature;
		
		/*
		 * Let's say we store the previous pcur-ppro and use THOSE To determine the right threshold!
		 * 
		 */
		float myacctmp;
		if(history_z == 0) {
			// start on the specified temperature
			myacctmp = acceptance_temperature; // 1.0;
		} else {
			// see what temp we'd need to get about the right acceptance rate (ignoring the log convexity)
			myacctmp = (history_sum / float(history_z)) / log(TARGET_ACCEPTANCE);
		}
		
		int swap = (is_valid(proposal->posterior) && random_float(RNG_ARGS) < exp( (ppro-pcur+fb)/myacctmp))
			|| is_invalid(current->posterior);
			
			
		// update history if it was a real swap
		if(is_valid(proposal->posterior) && is_valid(current->posterior) ) {
			if(history_z < HISTORY_SIZE) {
				history_z++;
			}
			else { 
				// update the running sum -- remove what will be overwritten
				// TODO: Not numerically stable for long runs...
				history_sum -= history[history_idx];
			}
			history[history_idx] = (ppro-pcur+fb);
			history_sum         += (ppro-pcur+fb);
			history_idx = (history_idx+1) % HISTORY_SIZE;
		}
			
		// swap if we should
		if(swap) {
				
			// swapadoo
			hypothesis* tmp=current; 
			current=proposal; 
			proposal=tmp;
			
			// update the chain acceptance count
			result->acceptance_count++;
						
			// and update the MAP if we should
			if(current->posterior > result->MAP.posterior || is_invalid(result->MAP.posterior)){
				COPY_HYPOTHESIS( &(result->MAP), current);
			}
		} // end if swap
		
	} // end main mcmc
		
	// and copy the sample to return
	COPY_HYPOTHESIS(&(result->sample), current);
	result->proposal_count += iterations;
	
}
 