/*
 * GPUropolis - 2013 Dec 15 - Steve Piantadosi 
 *
 * This adaptively increases the acceptance temperature as a function of how many rejections we've had in a row
 * to help chains move around (which appears to be a huge problem without this). 
 * 
 * It increases by a factor of INCREASE_PCT for each successive rejection greater than START_ADJUSTING_AT,
 * scaling the result (no matter what) by spec->acceptance_temperature
 * 
 * NOTE: This often has hypotheses jump around really similar ones. Maybe the temperature should be scaled to the likelihood? 
 * Or, we should jump 
 */

// MH kernel with just RR regeneration proposal (for simplicity)

// right choice of these can make you accept any proposal after some time in a rut...
__device__ const float INCREASE_SCALE = 1.01; 
__device__ const int START_ADJUSTING_AT = 0; // allow this many rejects in a row before we start to increase
__device__ const int FULL_RESAMPLE_AT = 9999999; // resample from the prior


__global__ void MH_adaptive_temperature_kernel(int N, mcmc_specification* all_spec, mcmc_results* all_results)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose
	
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
	
	// How many rejections in a row have we had?
	int sequential_rejection_count = 0; 
	float myacctmp = acceptance_temperature; // current temperature
	
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
		
		float pcur = current->prior/prior_temperature  + current->likelihood/likelihood_temperature;
		float ppro = proposal->prior/prior_temperature + proposal->likelihood/likelihood_temperature;
		
		// Either we fully re-draw from the prior
		// or we continue, adjusting temperature
		int swap = (is_valid(proposal->posterior) && random_float(RNG_ARGS) < exp( (ppro-pcur+fb)/myacctmp))
				|| is_invalid(current->posterior);
		
		// swap if we should
		if(swap) {
			// reset this counter if we actually move to a *new* hypothesis
			if(!dhypothesis_structurally_identical(current, proposal)) {
				myacctmp = acceptance_temperature;
				sequential_rejection_count = 0;
			}
			
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
		else {
			sequential_rejection_count++;
			
			// compute the acceptance temperature -- power increase for everything above START_ADJUSTING_AT
			if(sequential_rejection_count >= START_ADJUSTING_AT) 
				myacctmp = myacctmp * INCREASE_SCALE;	
		}
		
	} // end main mcmc
	
	result->proposal_count += iterations;
	
	// and yield this sample
	COPY_HYPOTHESIS(&(result->sample), current);
	
}
 