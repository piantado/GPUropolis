/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 *
 * Kernel for running tree-regeneration MCMC on CUDA
 * (insert/delete are not well debugged yet)
 */

// Kernel that executes on the CUDA device
// initialize_sample here will make us resample if 1; else we use out_hypothesis as-is and propose form that
// defaultly we use an annealing schedule: 1/2 of MCMC_ITERATIONS are spent going from LL_TEMP_START down to 1.0
// and the rest are spent at 1.0
__global__ void MH_kernel(int N, int PROPOSAL, int MCMC_ITERATIONS, float LL_TEMP_START, int DLEN, datum* device_data, hypothesis* out_hypotheses, hypothesis* out_MAPs, int myseed, int initialize_sample)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose, impossible to debug
	
	// Set up for the RNG
	int rx = idx+myseed;
	int ry = 362436069;
	int rz = 521288629;
	int rw = 88675123;
	
	data_t stack[MAX_MAX_PROGRAM_LENGTH+1]; // Create the stack, allowing for one constant for each op to be passed (really we only need half this, right?)
	
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
	update_hypothesis(current); // should go BEFORE compute_posterior
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
		for(int pp=0;pp<=3;pp++) { // for each kind of proposal
			float fb = 0.0;  // the forward-backward probability
			int ran=0; // did we actually run on this iteration, or were we looping over proposals not in PROPOSAL?
			
			if(pp==0 && (PROPOSAL & (0x1<<0)) ) { // full regeneration -- NOTE: THESE ARE ALMOST NEVER ACCEPTED SO WE NOW DON'T USE THEM!
				random_closed_expression(proposal, RNG_ARGS);
				
				// Update the proposal:
				update_hypothesis(proposal);
				
				fb += proposal->proposal_generation_lp;
				fb -= current->proposal_generation_lp;
				
				ran = 1;
			}
			else if(pp==1 && (PROPOSAL & (0x1<<1)) ) { // normal tree regen

				// generate a novel tree in program_buf
				random_closed_expression(tmpH1, RNG_ARGS);
				
				replace_random_subnode_with(current, proposal, tmpH1, RNG_ARGS);
				 
				// update the proposal:
				update_hypothesis(proposal);
				
				// forward-back probability
				fb += (nlogf(current->program_length)+proposal->proposal_generation_lp );
				fb -= (nlogf(proposal->program_length)+current->proposal_generation_lp );
				
				ran = 1;
			}
			else if(pp==2  && (PROPOSAL & (0x1<<2)) ) { // A generic insert move
			
				hypothesis* inserted = tmpH1; // the tree we insert. Note: An arbitrary subnode of this will be replaced with what's below the cut
				hypothesis* below_cut = tmpH2; // what's below the cut
				hypothesis* insertedWithBelow = tmpH3; // when we replace a node of inserted with below_cut, we put it here
				
				// create a random expression
				random_closed_expression(inserted, RNG_ARGS);
				
				// What subnode do we split on? Can't be the last
				int pos = dMAX_PROGRAM_LENGTH-1 - random_int(current->program_length-1, RNG_ARGS);
				
				// tmpH2 now holds the subtree of current
				copy_subnode_to( current, below_cut, pos );
						 
				// Now we put the pos subtree into a random place in tmpH1
				replace_random_subnode_with(inserted, insertedWithBelow, below_cut, RNG_ARGS);
				
				// and now we put all that mess back in current in the position we took it from
				replace_subnode_with(current, proposal, pos, insertedWithBelow);
				
				// and update
				update_hypothesis(proposal);
				
				// Handle forward/back with teh multiple path problem. 
				// TODO: CHECK THIS AGAIN PLEASE
				int cntidentical = count_identical_to(below_cut, insertedWithBelow, dMAX_PROGRAM_LENGTH-1); // how many subtrees of our replacement are equal to the bit we saved?
				
				// pick a node, generate rest of tree, and you could have replaced any of teh equivalent subnodes
				fb += nlogf(current->program_length-1) + (proposal->proposal_generation_lp - current->proposal_generation_lp)+logf(cntidentical);
				
				// to go back, pick the pos, pick any equivalent subnode, and propose
				fb -= nlogf(proposal->program_length-1) + logf(cntidentical) - logf(insertedWithBelow->program_length); // choose the right subnode to promote
				
				ran = 1;
			}
			else if(pp==3 && (PROPOSAL & (0x1<<2)) ) { // The inverse of the insert move -- take a random subtree and cut it out, promoting a child of X to X's position
				
				hypothesis* below = tmpH1; // we promote a child of this and put it in this guy's position
				hypothesis* belowbelow = tmpH2;
				
				// What subnode? Can't be the last
				int pos = dMAX_PROGRAM_LENGTH-1 - random_int(current->program_length-1, RNG_ARGS);
				
				// copy this subnode over
				copy_subnode_to( current, below, pos);
				
				// what node of this guy?
				int subpos = dMAX_PROGRAM_LENGTH-1 - random_int( below->program_length, RNG_ARGS);
				
				copy_subnode_to( below, belowbelow, pos);
				
				// and replace
				replace_subnode_with(current, proposal, pos, belowbelow);
				
				update_hypothesis(proposal);
				
				int cntid = count_identical_to(belowbelow, below, pos); 
				
				// Handle F/B -- TODO: CHECK THIS AGAIN PLEASE -- is there a way to promote a completely different node and end up with the same tree?
				fb += nlogf(current->program_length-1) + logf(cntid) - logf(below->program_length); // choose the right subnode to promote
				fb -= nlogf(proposal->program_length-1) + (current->proposal_generation_lp - proposal->proposal_generation_lp) + logf(cntid);	
		
				ran = 1;
			}
				
			if(ran) {
				// compute the posterior for the proposal
				compute_posterior(DLEN, device_data, proposal, stack);
				this_chain_proposals += 1; // how many total proposals have we made?
				
				// An annealing step. Since the prior is close to the generation probability, the difference in priors
				// will tend to cancel with fb. So we just do a temperature on the likelihood. 
				// For simplicity, we start a LL_TEMP_START and use half of our steps to go down to 1.0, and then 
				// the second half to sample at that likelihood temperature
				float lltemp = LL_TEMP_START * max(0.0, (1.0-float(mcmci*2)/float(MCMC_ITERATIONS))) + 1.0; // make the second half of steps at right temperature
				
				// compute whether not we accept the proposal, while rejecting infs and nans
				int swap = (random_float(RNG_ARGS) < exp( (proposal->prior + proposal->likelihood / lltemp - current->prior - current->likelihood/lltemp - fb) )) && is_valid(proposal->posterior) || !is_valid(current->posterior);
				
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
			} // end if ran
		} // end for each proposal kind
	}
	
	// And set the properties of current, and return:
	current->chain_index = idx; 
	current->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_hypotheses[idx], (void*)current, sizeof(hypothesis));
	
	current_MAP->chain_index = idx; 
	current_MAP->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_MAPs[idx], (void*)current_MAP, sizeof(hypothesis));
	
}
 