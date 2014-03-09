/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 *
 * Kernel for running tree-regeneration MCMC on CUDA
 * (insert/delete are not well debugged yet)
 */

// Kernel that executes on the CUDA device
// initialize_sample here will make us resample if 1; else we use out_hypothesis as-is and propose form that


// An adaptive annealing kernel -- increase the temperature for rejection and decrease for acceptance

__global__ void MH_kernel_AA(int N, int PROPOSAL, int MCMC_ITERATIONS, float LL_TEMP_START, int DLEN, datum* device_data, hypothesis* out_hypotheses, hypothesis* out_MAPs, int myseed, int initialize_sample)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose, impossible to debug
	
	// Set up for the RNG
	int rx = idx+myseed;
	int ry = 362436069;
	int rz = 521288629;
	int rw = 88675123;
	
	data_t stack[MAX_MAX_PROGRAM_LENGTH+1]; // Create the stack, allowing for one constant for each op to be passed (really we only need half this, right?)
	
	hypothesis current_, proposal_, tmpH1_, current_MAP_;
	hypothesis* current  = &current_;
	hypothesis* proposal = &proposal_;
	hypothesis* tmpH1  = &tmpH1_;
	hypothesis* current_MAP = &current_MAP_;
	
	// copy to most local, current
	memcpy((void*)current, &out_hypotheses[idx], sizeof(hypothesis));
	
	// randomly initialize if we should
	if(initialize_sample) random_closed_expression(current,  RNG_ARGS); 
	
	// update all the posterior, etc.
	update_hypothesis(current); // should go BEFORE compute_posterior
	compute_posterior(DLEN, device_data, current, stack);
	
	// initialize everything to be the current (especially proposal and MAP)
	memcpy( (void*) proposal,    (void*)current, sizeof(hypothesis));
	memcpy( (void*) current_MAP, (void*)current, sizeof(hypothesis));
	
	// Stats on MCMC -- here we start by smoothing with counts of 1
	int this_chain_acceptance_count = 0;
	int this_chain_proposals = 0;
	int current_acceptance_tries = 0;
	
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<MCMC_ITERATIONS;mcmci++) {

		// generate a novel tree in program_buf
		random_closed_expression(tmpH1, RNG_ARGS);
		
		replace_random_subnode_with(current, proposal, tmpH1, RNG_ARGS);
			
		// update the proposal:
		update_hypothesis(proposal);
		
		// forward-back probability
		float fb = 0.0;
		fb += (nlogf(current->program_length)+proposal->proposal_generation_lp );
		fb -= (nlogf(proposal->program_length)+current->proposal_generation_lp );
	
		// compute the posterior for the proposal
		compute_posterior(DLEN, device_data, proposal, stack);
		this_chain_proposals += 1; // how many total proposals have we made?
		
		// We'll scale exponentially between LL_TEMP_START and 1.0, using 0.5 - (the acceptace ratio) to interpolate between
		
		// exponentially increase the temperature as we reject
		float lltemp = __powf(LL_TEMP_START, float(current_acceptance_tries)/50.0); // divide by the target so when we = target, we are at T=1
		
// 		float r = float(this_chain_acceptance_count+1) / float(this_chain_proposals+1);
// 		float lltemp = __powf(LL_TEMP_START, 0.5-r); // divide by the target so when we = target, we are at T=1
		
		// compute whether not we accept the proposal, while rejecting infs and nans
		int swap = (random_float(rx,ry,rz,rw) < exp( (proposal->prior + proposal->likelihood / lltemp - current->prior - current->likelihood/lltemp - fb) )) && is_valid(proposal->posterior) || !is_valid(current->posterior);
		
		// swap if we should
		if(swap) {
			hypothesis* tmp;
			tmp = current;
			current  = proposal;
			proposal = tmp;
			
			// update the chain acceptance count
			this_chain_acceptance_count += 1;
			
			// reset counter for number of rejections in a row
			current_acceptance_tries = 0;
			
			// and update the MAP if we should
			if(current->posterior > current_MAP->posterior || !is_valid(current_MAP->posterior)){
				memcpy((void*)current_MAP, (void*)current, sizeof(hypothesis));
			}
		} else {
			current_acceptance_tries++;	
		}// end if swap
	
	}
	
	// And set the properties of current, and return:
	current->chain_index = idx; 
	current->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_hypotheses[idx], (void*)current, sizeof(hypothesis));
	
	current_MAP->chain_index = idx; 
	current_MAP->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_MAPs[idx], (void*)current_MAP, sizeof(hypothesis));
	
}
 