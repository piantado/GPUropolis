/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 *
 * Kernel for running tree-regeneration MCMC on CUDA
 * (insert/delete are not well debugged yet)
 */

// A Kernel for random hill-climbing search
__global__ void search_kernel(int N, int PROPOSAL, int MCMC_ITERATIONS, float POSTERIOR_TEMPERATURE, int DLEN, datum* device_data, hypothesis* out_hypotheses, hypothesis* out_MAPs, int myseed, int initialize_sample)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose, impossible to debug
	
	// Set up for the RNG
	int rx = idx+myseed;
	int ry = 362436069;
	int rz = 521288629;
	int rw = 88675123;
	
	data_t stack[MAX_MAX_PROGRAM_LENGTH+1]; // Create the stack, allowing for one constant for each op to be passed (really we only need half this, right?)
	
	hypothesis current_, proposal_, tmpH1_, tmpH2_, current_MAP_;
	hypothesis* current  = &current_;
	hypothesis* proposal = &proposal_;
	hypothesis* tmpH1    = &tmpH1_;
	hypothesis* tmpH2    = &tmpH2_;
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
	memcpy( (void*) tmpH1, (void*)current, sizeof(hypothesis));
	memcpy( (void*) tmpH2, (void*)current, sizeof(hypothesis));
	
	// Stats on MCMC
	int this_chain_acceptance_count = 0;
	int this_chain_proposals = 0;
	
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<MCMC_ITERATIONS;mcmci++) {
	
		float best_posterior = current->posterior; // since we start with proposal as current
		
		for(int proposali=0;proposali<100;proposali++) { // propose this many random expressions, and always accept the best
		
			// Just like a tree proposal above
			random_closed_expression(tmpH1, RNG_ARGS);
			replace_random_subnode_with(current, tmpH2, tmpH1, RNG_ARGS);
			
			// update the proposal:
			compute_length_and_proposal_generation_lp(tmpH2);
			
			compute_posterior(DLEN, device_data, tmpH2, stack);
			
			if(tmpH2->posterior > best_posterior && is_valid(tmpH2->posterior)) {
				best_posterior = tmpH2->posterior;
				hypothesis* t = proposal;
				proposal = tmpH2;
				tmpH2 = t;
			}
		}
			
		// and update the MAP if we should
		if(current->posterior > current_MAP->posterior || !is_valid(current_MAP->posterior)){
			memcpy((void*)current_MAP, (void*)current, sizeof(hypothesis));
		}
			
	}
	
	// And set the properties of current, and return:
	current->chain_index = idx; 
	current->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_hypotheses[idx], (void*)current, sizeof(hypothesis));
	
	current_MAP->chain_index = idx; 
	current_MAP->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_MAPs[idx], (void*)current_MAP, sizeof(hypothesis));
	
}
 