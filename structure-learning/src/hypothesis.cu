/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Hypothesis definitions and functions
 */


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Set program length and number of constants
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define MAX_PROGRAM_LENGTH 10
const int MAX_CONSTANTS = 10; // how many constants per hypothesis at most?

enum CONSTANT_TYPES { GAUSSIAN, EXPONENTIAL, UNIFORM,   __N_CONSTANT_TYPES};
// enum CONSTANT_TYPES { GAUSSIAN,        __N_CONSTANT_TYPES};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// what type is the program?
typedef char op_t;

#define CHECK_BIT -3453

// A struct for storing a hypothesis, which is a program and some accoutrements
// the check variables are there to make sure we don't overrun anything in any program editing
// main.cu has assertions to check they have CHECK_BIT value
// This also stores some ability for constants, but these are not used by all MH schemes
typedef struct hypothesis {
	float prior;
	int check6;
	float likelihood;
	int check0;
	float posterior;
	int check1;
	float temperature;
	float proposal_generation_lp; // with what log probability does our proposal mechanism generate this? 
	int program_length; // how long is my program?
	int check2;
	op_t   program[MAX_PROGRAM_LENGTH];
	int check3;
	int chain_index; // what index am I?
	int check4;
	float constants[MAX_CONSTANTS];
	int check5;
	int constant_types[MAX_CONSTANTS]; // 
	int nconstants; // how many constants are used?
	int found_count; // how many times has this been found?
} hypothesis;

#define COPY_HYPOTHESIS(x,y) memcpy( (void*)(x), (void*)(y), sizeof(hypothesis));
#define COPY_CONSTANTS(x,y)  memcpy( (void*)((x)->constants), (void*)((y)->constants), sizeof(int)*MAX_CONSTANTS);


// A standard initialization that sets this thing up!
void initialize(hypothesis* h, RNG_DEF){
	h->posterior = -1.0/0.0;
	h->prior = -1.0/0.0;
	h->likelihood = 0.0;
	h->proposal_generation_lp = 0.0; 
	h->program_length = 0.0;
	h->nconstants = 0;
	h->found_count = 0;
	h->chain_index = -1;
	
	// Set some check bits to catch buffer overruns. If any change, we have a problem!
	h->check0 = CHECK_BIT;	h->check1 = CHECK_BIT;	h->check2 = CHECK_BIT;	h->check3 = CHECK_BIT; 	h->check4 = CHECK_BIT; 	h->check5 = CHECK_BIT; 	h->check6 = CHECK_BIT;
	
	// zero the program
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++)
		h->program[i] = NOOP_; 
	
	// Set up our constants -- just random
	for(int i=0;i<MAX_CONSTANTS;i++) {
		h->constants[i] = random_normal(RNG_ARGS);
		h->constant_types[i] = GAUSSIAN;
	}
}

// so we qsort so best is LAST
int hypothesis_posterior_compare(const void* a, const void* b) {
	float ap = ((hypothesis*)a)->posterior;
	float bp = ((hypothesis*)b)->posterior; 
	
	// use != here to check for nan:
	if( ap>bp || (bp!=bp) ) { return 1; }
	if( ap<bp || (ap!=ap) ) { return -1; }
	return 0;
}

// so we qsort so best is FIRST
int neghypothesis_posterior_compare(const void* a, const void* b) {
	return -hypothesis_posterior_compare(a,b);
}

// sort so that we can remove duplicate hypotheses
int sort_bestfirst_unique(const void* a, const void* b) {
	int c = neghypothesis_posterior_compare(a,b);
	
	if(c != 0) return c;
	else { // we must sort by the program (otherwise hyps with identical posteriors may not be removed as duplicates)
		hypothesis* ah = (hypothesis*) a;
		hypothesis* bh = (hypothesis*) b;
		for(int i=MAX_PROGRAM_LENGTH-1;i>0;i--){
			op_t x = ah->program[i]; op_t y = bh->program[i];
			if(x<y) return 1;
			if(x>y) return -1;
		}
		
		// deal with constants
		for(int i=0;i<MAX_CONSTANTS;i++){
			float x = ah->constants[i];
			float y = bh->constants[i];
			if(x<y) return 1;
			if(x>y) return -1;
		}
		
		for(int i=0;i<MAX_CONSTANTS;i++){
			int x = ah->constant_types[i];
			int y = bh->constant_types[i];
			if(x<y) return 1;
			if(x>y) return -1;
		}	
	}
	
	return 0;
}



__host__ int hypothesis_structurally_identical( hypothesis* a, hypothesis* b) {
	// check if two hypotheses are equal in terms of the program (ignoring constants and other params)
	
	if(a->program_length != b->program_length) return 0;
	
	// now loop back, only up to program_length
	for(int i=MAX_PROGRAM_LENGTH-1;i>=MAX_PROGRAM_LENGTH-a->program_length;i--) {
		if(a->program[i] != b->program[i]) return 0;
	}
	return 1;
}// UUGH WE JUST REPEAT BELOW B/C MAX_PROGRAM_LENGTH IS DIFFERENT BETWEEN HOST AND DEVICE
__device__ int dhypothesis_structurally_identical( hypothesis* a, hypothesis* b) { 
	// check if two hypotheses are equal in terms of the program (ignoring constants and other params)
	
	if(a->program_length != b->program_length) return 0;
	
	// now loop back, only up to program_length
	for(int i=MAX_PROGRAM_LENGTH-1;i>=MAX_PROGRAM_LENGTH-a->program_length;i--) {
		if(a->program[i] != b->program[i]) return 0;
	}
	return 1;
}


// void print_program_as_expression(hypothesis* h); // needed by print but defined in programs.cu
// void print_hypothesis(FILE* fp, int outer, int rank, int print_stack, hypothesis* h){
// 
// 	fprintf(fp, "%d\t%d\t%d\t%f\t%4.4f\t%4.4f\t%4.4f\t", ismap, outer, rank, h->chain_index, h->posterior,  h->prior, h->likelihood, h->acceptance_ratio);
// 	
// 	if(print_stack){
// 		printf("\"");
// 		for(int i=0;i<MAX_PROGRAM_LENGTH;i++) 
// 			printf("%d ", h->program[i]);
// 		printf("\"\t");
// 	}
// 	
// 	printf("\"");
// 	print_program_as_expression( h );
// 	printf("\"\n");
// }

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Bayes on hypotheses
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// These are used below but defined for real elsewhere
__device__ float f_output(float x, hypothesis* h, float* stack);
__device__ float compute_x1depth_prior(hypothesis* h);

__device__ float compute_likelihood(int DLEN, datum* device_data, hypothesis* h, float* stack) {

	float ll = 0.0;
	for(int i=0;i<DLEN;i++){
		//__syncthreads(); // NOT a speed improvement
		
		float val = f_output( device_data[i].input, h, stack);
		
		// compute the difference between the output and what we see
		data_t d = device_data[i].output - val;
		
		// and use a gaussian likelihood
		ll += lnormalpdf(d, device_data[i].sd);
	}
	
	// ensure valid
	if (!is_valid(ll)) ll = -1.0f/0.0f;
	
 	h->likelihood = ll / LL_TEMPERATURE;
	
	return h->likelihood;
	
}

__device__ float compute_constants_prior(hypothesis* h);
__device__ float compute_prior(hypothesis* h) {
	
	// and check that we're not too long (since that leads to problems, apparently even if we are exactly the right length)
	if(h->program_length >= MAX_PROGRAM_LENGTH){
		h->prior = -1.0f/0.0f;	
		return h->prior;
	}
	
	float prior = 0.0;	
	
	// We just use the proposal as a prior
	// NOTE: This means compute_generation_probability MUST be called beforee compute_posterior
	//prior = compute_generation_probability(h); // the prior just on the program
 	prior +=  h->proposal_generation_lp / PRIOR_TEMPERATURE;
	
	// The fancy other prior
// 	prior += compute_x1depth_prior(h); // only apply PRIOR_TEMPERATURE to the PCFG part, not to this X part
	
	// Compute the constant prior (if we used constants)
	prior += compute_constants_prior(h);	
	
	h->prior = prior;
	return h->prior;
}

__device__ void compute_posterior(int DLEN, datum* device_data, hypothesis* h, float* stack) {
	compute_prior(h);
	compute_likelihood(DLEN, device_data, h, stack);
	h->posterior = h->prior + h->likelihood;
}



