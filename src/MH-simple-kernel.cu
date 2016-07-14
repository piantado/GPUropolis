/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 *
 * Simple kernel for running tree-regeneration MCMC on CUDA
 * 
 */

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Proposals
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// make a proposal from from to to, 
__device__ float propose_rewrite(hypothesis* to, hypothesis* from, RNG_DEF){
    
    COPY_HYPOTHESIS( to, from );
    
    // insert between start and stop, shifting by shift
    int start = random_int(MAX_PROGRAM_LENGTH-1, RNG_ARGS);
    int stop  = start+random_int(MAX_PROGRAM_LENGTH-start, RNG_ARGS); 
    
    for(int p=start;p<stop;p++){
        to->program[p] = random_op(RNG_ARGS);
    }
    
    return 0.0;
}

__device__ float propose_insert(hypothesis* to, hypothesis* from, RNG_DEF){
    
    COPY_HYPOTHESIS( to, from );
    
    // insert between start and stop, shifting to the right
    int start = random_int(MAX_PROGRAM_LENGTH, RNG_ARGS); // TODO: Check off by 1?
    int stop  = start+random_int(MAX_PROGRAM_LENGTH-start, RNG_ARGS); 
    int l = stop-start;
    
    // shift after stop
    for(int p=start;p<MAX_PROGRAM_LENGTH-l;p++){
        to->program[p+l] = to->program[p];
        to->program[p] = random_op(RNG_ARGS);
    }
     
    // both forward and back have to choose start and stop; 
    // but forward has a (1/NUM_OPS)**l probability too 
    return -l*logf(NUM_OPS); 
}

__device__ float propose_delete(hypothesis* to, hypothesis* from, RNG_DEF){
    
    COPY_HYPOTHESIS( to, from );
    
    // insert between start and stop, shifting to the right
    int start = random_int(MAX_PROGRAM_LENGTH, RNG_ARGS); // TODO: Check off by 1?
    int stop  = start+random_int(MAX_PROGRAM_LENGTH-start, RNG_ARGS); 
    int l = stop-start;
    
    // shift after stop
    for(int p=start;p<MAX_PROGRAM_LENGTH-l;p++){
        to->program[p] = to->program[p+1];  
    }
    // TODO: Hmm what to do with the deleted end of to? Leave as-is?
     
    // both forward and back have to choose start and stop; 
    // but backward has a (1/NUM_OPS)**l generation probability
    return l*logf(NUM_OPS); 
}

__device__ float propose_constants(hypothesis* to, hypothesis* from, RNG_DEF){
    
    COPY_HYPOTHESIS( to, from );
  
    // we'll propose a random scaling factor increase or decrease
    float r = random_float(RNG_ARGS); // percent change up or down
    
    int k = random_int(MAX_CONSTANTS, RNG_ARGS);

    if(random_int(2,RNG_ARGS) == 1) {
        to->constants[k] = from->constants[k]*(1.0+r*0.01);
    }
    else {
        to->constants[k] = from->constants[k]/(1.0+r*0.01);
    }
       
    return 0.0;
}

__device__ float propose_constant_types(hypothesis* to, hypothesis* from, RNG_DEF){
    
    COPY_HYPOTHESIS( to, from );
    
    int k = random_int(MAX_CONSTANTS, RNG_ARGS);
    to->constant_types[k] = random_int(__N_CONSTANT_TYPES, RNG_ARGS);
    
    return 0.0;
}
   
// just resample from the prior
__device__ float prior_propose(hypothesis* to, hypothesis* from, RNG_DEF){
    
    for(int i=0;i<MAX_PROGRAM_LENGTH;i++)
        to->program[i] = random_op(RNG_ARGS); 
    
    // randomize constants and their types
    randomize_constants(to, RNG_ARGS);
    
    
    // now we have to get the forward - backward
    float fb = 0.0;
    for(int k=0;k<MAX_CONSTANTS;k++) {
        fb += lnormalpdf(to->constants[k], 1.0) - lnormalpdf(from->constants[k], 1.0);
    }
    
    return fb;
}
   


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MCMC Kernel
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__global__ void MH_simple_kernel(int N, mcmc_specification* all_spec, mcmc_results* all_results)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose
	
	// get my own spec and results
	mcmc_specification* spec   = &(all_spec[idx]);
	mcmc_results*       result = &(all_results[idx]); 
	
	// unpackage the specification
	int iterations               = spec->iterations;	
	float acceptance_temperature = spec->acceptance_temperature;
	int data_length              = spec->data_length;
	datum* data                  = spec->data;
    data_t* stack                = spec->stack; // stack for processing
	
	// Set up for the RNG
 	int rx = spec->rng_seed;
	
    hypothesis* current  = &result->sample; // store this in result
    hypothesis* proposal = &spec->proposal;
	
	current->chain_index = idx;	// which chain did you come from?
    
	compute_posterior(data_length, data, current, stack);
    
	// initialize everything to be the current
	COPY_HYPOTHESIS( &(result->sample), current );
	COPY_HYPOTHESIS( &(result->MAP),    current );
	
	// Stats on MCMC
	result->acceptance_count = 0;
	result->proposal_count = 0;
	result->rng_seed = spec->rng_seed; // store this
        
	// ---------------------------------------------------------------------------------
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<iterations;mcmci++) {
		
		float fb = 0.0; // forward minus backward probability
        
        // use a switch case to set the probability of each kind of proposal
        // NOTE: This means if we just run 1 or 2 steps, we will only get structural proposals
        switch(mcmci%6){
            case 0:
                fb = propose_rewrite(proposal, current, RNG_ARGS);
                break;
            case 1:
                fb = propose_insert(proposal, current, RNG_ARGS);
                break;
            case 2:
                fb = propose_delete(proposal, current, RNG_ARGS);
                break;
            case 3:
                fb = propose_constants(proposal, current, RNG_ARGS);
                break;
            case 4:
                fb = propose_constant_types(proposal, current, RNG_ARGS);
                break;
            case 5:
                fb = prior_propose(proposal, current, RNG_ARGS);
                break;
        }

        // compute the posterior (setting prior, likelihood on proposal)
 		compute_posterior(data_length, data, proposal, stack);
		
		int swap = (is_valid(proposal->posterior) && 
		           random_float(RNG_ARGS) < exp( (proposal->posterior-current->posterior-fb)/acceptance_temperature ))
			   || is_invalid(current->posterior);
	
		if(swap) { // swap if we should
			
			hypothesis *tmp=current; // no memory allocation
			current=proposal; 
			proposal=tmp;
			
			// update the chain acceptance count
			result->acceptance_count++;
			
			// and update the MAP if we should
			if(current->posterior > result->MAP.posterior || is_invalid(result->MAP.posterior)){
 				COPY_HYPOTHESIS( &(result->MAP), current);
			}
		} // end if swap*/
		
	} // end main mcmc
	
	result->proposal_count += iterations;
	
	// and copy over if current is pointing to our old proposal
    if(current != &(result->sample))
        COPY_HYPOTHESIS(&(result->sample), current);
	
}
 