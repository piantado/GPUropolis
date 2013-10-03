/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 *
 * Kernel for running simple generation from the prior
 */

// Kernel that executes on the CUDA device
// initialize_sample here will make us resample if 1; else we use out_hypothesis as-is and propose form that
__global__ void prior_kernel(int N, int PROPOSAL, int MCMC_ITERATIONS, float POSTERIOR_TEMPERATURE, int DLEN, datum* device_data, hypothesis* out_hypotheses, hypothesis* out_MAPs, int myseed, int initialize_sample)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose, impossible to debug
	
	// Set up for the RNG
	int rx = idx+myseed;
	int ry = 362436069;
	int rz = 521288629;
	int rw = 88675123;
	
	data_t stack[MAX_MAX_PROGRAM_LENGTH+1]; // Create the stack, allowing for one constant for each op to be passed (really we only need half this, right?)
	
	hypothesis current_, proposal_, current_MAP_;
	hypothesis* current  = &current_;
	hypothesis* proposal = &proposal_;
	hypothesis* current_MAP = &current_MAP_;
	
	// copy to most local, current
	memcpy((void*)current, &out_hypotheses[idx], sizeof(hypothesis));
	
	// randomly initialize if we should
	if(initialize_sample) random_closed_expression(current,  RNG_ARGS); 
	
	// update all the posterior, etc.
	compute_length_and_proposal_generation_lp(current); // should go BEFORE compute_posterior
	compute_posterior(DLEN, device_data, current, stack);
	
	// initialize everything to be the current (especially proposal and MAP)
	memcpy( (void*) proposal,    (void*)current, sizeof(hypothesis));
	memcpy( (void*) current_MAP, (void*)current, sizeof(hypothesis));
	
	// Stats on MCMC
	int this_chain_acceptance_count = 0;
	int this_chain_proposals = 0;
	
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<MCMC_ITERATIONS;mcmci++) {
			float fb = 0.0;  // the forward-backward probability

			random_closed_expression(proposal, RNG_ARGS);
				
			// Update the proposal:
			compute_length_and_proposal_generation_lp(proposal);
			
			fb += proposal->proposal_generation_lp;
			fb -= current->proposal_generation_lp;
			
			// compute the posterior for the proposal
			compute_posterior(DLEN, device_data, proposal, stack);
			this_chain_proposals += 1; // how many total proposals have we made?
						
			// compute whether not we accept the proposal, while rejecting infs and nans
			int swap = (random_float(RNG_ARGS) < exp((proposal->posterior - current->posterior)/POSTERIOR_TEMPERATURE - fb) && is_valid(proposal->posterior)) || !is_valid(current->posterior);
			
			// swap if we should
			if(swap) {
				hypothesis* tmp;
				tmp = current;
				current  = proposal;
				proposal = tmp;
				
				// update the chain acceptance count
				this_chain_acceptance_count += 1;
				
				// and update the MAP if we should
				if(current->posterior > current_MAP->posterior || !is_valid(current_MAP->posterior)){
					memcpy((void*)current_MAP, (void*)current, sizeof(hypothesis));
				}
			} // end if swap
				
	}
	
	// And set the properties of current, and return:
	current->chain_index = idx; 
	current->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_hypotheses[idx], (void*)current, sizeof(hypothesis));
	
	current_MAP->chain_index = idx; 
	current_MAP->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_MAPs[idx], (void*)current_MAP, sizeof(hypothesis));
	
}
 