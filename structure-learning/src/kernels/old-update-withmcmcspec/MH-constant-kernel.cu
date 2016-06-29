/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 *
 * Kernel for running tree-regeneration MCMC on CUDA
 * (insert/delete are not well debugged yet)
 */

// A simplified version of MH_kernel that also does updates to the constant array
__global__ void MH_constant_kernel(int N, int MCMC_ITERATIONS, int DLEN, datum* device_data, hypothesis* out_hypotheses, hypothesis* out_MAPs, int myseed, int initialize_sample)
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
	hypothesis* current     = &current_;
	hypothesis* proposal    = &proposal_;
	hypothesis* tmpH1       = &tmpH1_;
	hypothesis* tmpH2       = &tmpH2_;
	hypothesis* tmpH3       = &tmpH3_;
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
	memcpy( (void*) tmpH1,       (void*)current, sizeof(hypothesis));
	memcpy( (void*) tmpH2,       (void*)current, sizeof(hypothesis));
	memcpy( (void*) tmpH3,       (void*)current, sizeof(hypothesis));
	
	// Stats on MCMC
	int this_chain_acceptance_count = 0;
	int this_chain_proposals = 0;
	
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<MCMC_ITERATIONS;mcmci++) {
		for(int pp=1;pp<=3;pp++) { // for each kind of proposal -- NOTE: WE NOW SKIP THE CONST
			
			float fb = 0.0;  // the forward-backward probability
			
			if(pp==0) {
				
				// a full proposal from the prior (to extract us from local max)
				random_closed_expression(proposal, RNG_ARGS);
				update_hypothesis(proposal); // sets the number of constants
				
				// and set the constants to new random values
				// and compute fb!
				for(int k=0;k<proposal->nconstants;k++) {
					float old_val = proposal->constants[k];
					float new_val = random_normal(RNG_ARGS);
					fb += lnormalpdf(new_val,1.0) - lnormalpdf(old_val, 1.0);
					
					proposal->constants[k] = new_val;
					proposal->constant_types[k] = random_int(__N_CONSTANT_TYPES, RNG_ARGS);
					
				}
				
				fb += proposal->proposal_generation_lp;
				fb -= current->proposal_generation_lp;
				
			}			
			if(pp==1) { // normal tree regen

				// generate a novel tree in program_buf
				random_closed_expression(tmpH1, RNG_ARGS);
				
				replace_random_subnode_with(current, proposal, tmpH1, RNG_ARGS);
				 
				// update the proposal:
				update_hypothesis(proposal);
				
				// forward-back probability
				fb += (nlogf(current->program_length)+proposal->proposal_generation_lp );
				fb -= (nlogf(proposal->program_length)+current->proposal_generation_lp );
			}
			else if(pp==2) { // constant proposal
				
				// copy over
				memcpy((void*)proposal, (void*)current, sizeof(hypothesis));
				
				fb += resample_random_constant(proposal, RNG_ARGS);
				
				update_hypothesis(proposal); // need to do this to recompute prior
			}
			else if(pp==3) { // constant type proposal
				
				// copy over
				memcpy((void*)proposal, (void*)current, sizeof(hypothesis));
				
				fb += resample_random_constant_type(proposal, RNG_ARGS);
				
				update_hypothesis(proposal); // need to do this to recompute prior
			}
			
			// compute the posterior for the proposal
			compute_posterior(DLEN, device_data, proposal, stack);
			this_chain_proposals += 1; // how many total proposals have we made?
						
			// compute whether not we accept the proposal, while rejecting infs and nans
			int swap = random_float(RNG_ARGS) < exp( proposal->posterior - current->posterior - fb) && is_valid(proposal->posterior);
			
			// swap if we should
			if(swap) {
				hypothesis* tmp;
				tmp = current;
				current  = proposal;
				proposal = tmp;
				
				// update the chain acceptance count
				this_chain_acceptance_count += 1;
				
				// and update the MAP if we should
				if(current->posterior > current_MAP->posterior || is_invalid(current_MAP->posterior)){
					memcpy((void*)current_MAP, (void*)current, sizeof(hypothesis));
				}
			} // end if swap
		
		} // end for each proposal kind
	}
	
	// And set the properties of current, and return:
// 	current->posterior = my_pow(-7.589611, 2.3);
	current->chain_index = idx; 
	current->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_hypotheses[idx], (void*)current, sizeof(hypothesis));
	
// 	current_MAP->posterior = my_pow(-7.589611, 2.0);
	current_MAP->chain_index = idx; 
	current_MAP->acceptance_ratio = float(this_chain_acceptance_count)/float(this_chain_proposals);
	memcpy( &out_MAPs[idx], (void*)current_MAP, sizeof(hypothesis));
	
}
 