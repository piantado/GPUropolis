/*
 * GPUropolis - 2013 Dec 15 - Steve Piantadosi 
 *
 * This tries to detect local maxima by looking for a small variance in jump size. When it finds one, it resumes 
 * with a draw from the prior.
 * 
 * 
 */

// MH kernel with just RR regeneration proposal (for simplicity)

// right choice of these can make you accept any proposal after some time in a rut...
__device__ const float THRESHOLD = 0.001; // this percentage within; decreasing makes it harder to reset
__device__ const int ALLOWED_STEPS_WITHIN_THRESHOLD = 1000; // if we take this many steps within THRESHOLD of 


__global__ void MH_detect_local_maxima(int N, mcmc_specification* all_spec, mcmc_results* all_results)
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
	
	// How many steps have been within our threshold for local max?
	int steps_within_threshold = 0; 
	float old_lp = -9.e99;
	
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
		
		// Either we fully re-draw from the prior
		// or we continue, adjusting temperature
		if(steps_within_threshold > ALLOWED_STEPS_WITHIN_THRESHOLD) {
			random_closed_expression(current,  RNG_ARGS); 	
			update_hypothesis(current); // should go BEFORE compute_posterior
			compute_posterior(data_length, data, current, stack);
			steps_within_threshold = 0;
		}
		else {
		
			int swap = (is_valid(proposal->posterior) && random_float(RNG_ARGS) < exp( (ppro-pcur+fb)/acceptance_temperature))
				|| is_invalid(current->posterior);
		
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
			
			if( abs(old_lp - current->posterior) > THRESHOLD) {
				steps_within_threshold = 0;
			}
			else {
				steps_within_threshold++;
			}
		}
		
		old_lp = current->posterior;
	} // end main mcmc
	
	result->proposal_count += iterations;
	
// 	current->prior = proposal->posterior;
	
	// and yield this sample
	COPY_HYPOTHESIS(&(result->sample), current);
	
}
 