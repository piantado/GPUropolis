/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 *
 * Kernel for running tree-regeneration MCMC on CUDA
 * (insert/delete are not well debugged yet)
 */

// Kernel that executes on the CUDA device
// a version that weights subtrees by their depth in sampling, so that we can preferentially resample low leaves 
// for hopefully a higher acceptance rate

__global__ void MH_weighted_kernel(int N, int MCMC_ITERATIONS, int DLEN, datum* device_data, hypothesis* out_hypotheses, hypothesis* out_MAPs, int myseed, int initialize_sample)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose, impossible to debug
	
	// Set up for the RNG
	int rx = idx+myseed;
	int ry = 362436069;
	int rz = 521288629;
	int rw = 88675123;
	
	data_t stack[MAX_MAX_PROGRAM_LENGTH+1]; // Create the stack, allowing for one constant for each op to be passed (really we only need half this, right?)
	
	int depth_counts[MAX_MAX_PROGRAM_LENGTH]; // how many primitives are at each depth?
	
	hypothesis current_, proposal_, tmpH1_, tmpH2_, tmpH3_, current_MAP_;
	hypothesis* current  = &current_;
	hypothesis* proposal = &proposal_;
	hypothesis* tmpH1    = &tmpH1_;
	hypothesis* tmpH2    = &tmpH2_;
	hypothesis* tmpH3    = &tmpH3_;
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
	memcpy( (void*) tmpH3, (void*)current, sizeof(hypothesis));
	
	// Stats on MCMC
	int this_chain_acceptance_count = 0;
	int this_chain_proposals = 0;
	
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<MCMC_ITERATIONS;mcmci++) {
		float fb = 0.0;  // the forward-backward probability
		
		// generate a novel tree in program_buf
		random_closed_expression(tmpH1, RNG_ARGS);
		
		// instead of using replace_random_subnode_with, we here do soem fanciness with the depth
		// first count the number at each depth
		
		// zero the array
		for(int i=0;i<dMAX_PROGRAM_LENGTH;i++) depth_counts[i] = 0;
		
		// count how many at each depth:
		int nopen = 1;	
		int max_depth = 0;
		for(int i=dMAX_PROGRAM_LENGTH-1;i>=0 && nopen != 0;i--) {
			nopen -= stack_change(current->program[i]);
			depth_counts[nopen]++;
			if(nopen > max_depth) max_depth = nopen;
		}
		
		float P = 0.8; // the probability of moving up one level in picking a node to propose
		
		// sample a depth from a geometric between 1..max_depth, inclusive
		int d = max_depth - truncated_geometric(P, max_depth, RNG_ARGS);
		
		while(depth_counts[d]==0) d = max_depth - truncated_geometric(P, max_depth, RNG_ARGS); // and fix the problem that if we run off the end of the program length, we might have zero counts for this depth:
		
		assert(depth_counts[d] > 0);
		assert(d >= 0 && d <= max_depth); // just be sure
		
		// So a weird corner case is when we don't finish the expression (it runs to dMAX_PROGRAM_LENGTH 
		// before terminating, and we sample 0. So if the counts are 0, we'll just move to the next sample
// 		if( depth_counts[d] == 0) { continue; } 
		
		// and choose *which* node with this depth
		int di = random_int(depth_counts[d], RNG_ARGS);
		// and find his position
		int pos = -1;
		nopen = 1; // reset
		for(int i=dMAX_PROGRAM_LENGTH-1;i>=0  && nopen != 0;i--) {
			nopen -= stack_change(current->program[i]);
			if(nopen == d) { // if we're at the target depth
				if(di==0) pos = i; // see if we're the di'th node
				di--;
			}
		}
		// TODO IMMEDATELY: WHY DOES THIS ASSERTION FAIL?
// 		assert(pos != -1); // now pos should store a random node at that depth		
		
		// so replace:
		replace_subnode_with(current, proposal, pos, tmpH1);
			
		// update the proposal:
		compute_length_and_proposal_generation_lp(proposal);
		
		// and the backward -- how many in proposal are at that depth?
		int p_d = 0;
		int p_max_depth = 0;
		nopen = 1; // reset
		for(int i=dMAX_PROGRAM_LENGTH-1;i>=0  && nopen != 0;i--) { // find the max depth and the number at depth d
			nopen -= stack_change(proposal->program[i]);
			if(nopen == d) p_d++;
			if(nopen > p_max_depth) p_max_depth = nopen;
		}
		
		fb +=  ltruncated_geometric(max_depth-d, P, max_depth); + nlogf(depth_counts[d]) + proposal->proposal_generation_lp;
		fb -=  ltruncated_geometric(p_max_depth-d, P, p_max_depth); + nlogf(p_d) + current->proposal_generation_lp;
		
		// compute the posterior for the proposal
		compute_posterior(DLEN, device_data, proposal, stack);
		this_chain_proposals += 1; // how many total proposals have we made?
		
		// compute whether not we accept the proposal, while rejecting infs and nans
		int swap = (random_float(RNG_ARGS) < exp( (proposal->prior + proposal->likelihood - current->prior - current->likelihood - fb) )) && is_valid(proposal->posterior) || !is_valid(current->posterior);
		
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
 