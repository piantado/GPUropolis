/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Hypothesis definitions and functions
 */


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Functions for setting and manipulating the program length
// here all hypotheses are allocated the maximum amount of memory
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

const int      MAX_MAX_PROGRAM_LENGTH = 100; // the most this could ever be. 
int            hMAX_PROGRAM_LENGTH = 25; // on the hostA
__device__ int dMAX_PROGRAM_LENGTH = 25; // on the device

// we must use a function to set the program length, because its used on local and global vars
// TODO: WHEN WE DO THIS, WE HAVE TO MAKE SURE THE HYPOTHESES ARE COPIED TO THE END OF THEIR PROGRAMS IF WE WANT THEM!
void set_MAX_PROGRAM_LENGTH(int v){
	assert(v < MAX_MAX_PROGRAM_LENGTH);
	hMAX_PROGRAM_LENGTH = v;
	cudaMemcpyToSymbol(dMAX_PROGRAM_LENGTH,&hMAX_PROGRAM_LENGTH,sizeof(int));
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Variables for the constants and their types
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

const int      MAX_CONSTANTS = 5; // how many constants per hypothesis at most?

enum CONSTANT_TYPES { GAUSSIAN, LOGNORMAL, EXPONENTIAL, UNIFORM,        __N_CONSTANT_TYPES};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// what type is the program?
typedef char op_t;

#define CHECK_BIT 33

// A struct for storing a hypothesis, which is a program and some accoutrements
// the check variables are there to make sure we don't overrun anything in any program editing
// main.cu has assertions to check they have CHECK_BIT value
// This also stores some ability for constants, but these are not used by all MH schemes
typedef struct hypothesis {
	float prior;
	float likelihood;
	int check0;
	float posterior;
	int check1;
	float temperature;
	float proposal_generation_lp; // with what log probability does our proposal mechanism generate this? 
	int program_length; // how long is my program?
	float acceptance_ratio; // after we run, what's the acceptance ratio for each thread?
	int check2;
	op_t   program[MAX_MAX_PROGRAM_LENGTH];
	int check3;
	int chain_index; // what index am I?
	int check4;
	float constants[MAX_CONSTANTS];
	int check5;
	int constant_types[MAX_CONSTANTS]; // 
	int check6;
	int nconstants; // how many constants are used?
// 	int mutable_end; // what is the last index of program that proposals can change?
} hypothesis;




// A standard initialization that sets this thing up!
void initialize(hypothesis* h){
	h->posterior = -1.0/0.0;
	h->prior = -1.0/0.0;
	h->likelihood = 0.0;
	h->proposal_generation_lp = 0.0; 
	h->program_length = 0.0;
	h->acceptance_ratio = 0.0;
	h->nconstants = 0;

	// Set some check bits to catch buffer overruns. If any change, we have a problem!
	h->check0 = CHECK_BIT;	h->check1 = CHECK_BIT;	h->check2 = CHECK_BIT;	h->check3 = CHECK_BIT; 	h->check4 = CHECK_BIT; 	h->check5 = CHECK_BIT; 	h->check6 = CHECK_BIT;
	
	// zero the program
	for(int i=0;i<hMAX_PROGRAM_LENGTH;i++)
		h->program[i] = NOOP_; 
	
	// zero our constants
	for(int i=0;i<MAX_CONSTANTS;i++) {
		h->constants[i] = 0.0f;
		h->constant_types[i] = 0x0;
	}
}

// so we qsort so best is LAST
int hypothesis_posterior_compare(const void* a, const void* b) {
	float ap = ((hypothesis*)a)->posterior;
	float bp = ((hypothesis*)b)->posterior; 
	
	if( ap>bp ) { return 1; }
	if( ap<bp ) { return -1; }
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
		for(int i=hMAX_PROGRAM_LENGTH-1;i>0;i--){
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



int hypothesis_structurally_identical( hypothesis* a, hypothesis* b) {
	// check if two hypotheses are equal in terms of the program (ignoring constants and other params)
	
	if(a->program_length != b->program_length) return 0;
	
	// now loop back, only up to program_length
	for(int i=hMAX_PROGRAM_LENGTH-1;i>=hMAX_PROGRAM_LENGTH-a->program_length;i--) {
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
// Manipulate constants in hypotheses
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Take h and resample one of its constants according to the types
// and return the f/b probability
// __device__ float resample_random_constant(hypothesis* h, RNG_DEF) {
// 	
// 	int k = random_int(MAX_CONSTANTS, RNG_ARGS);
// 	
// 	float old_value = h->constants[k];
// 	float new_value = 0.0;
// 	float fb = 0.0;
// 	switch(h->constant_types[k]) {
// 		case GAUSSIAN:
// 			new_value = random_normal(RNG_ARGS);
// 			fb = lnormalpdf(new_value, 1.0) - lnormalpdf(old_value, 1.0); // these and all constants are set to the generation defaults
// 			break;
// 		case LOGNORMAL:
// 			new_value = random_lnormal(0.0, 1.0, RNG_ARGS);
// 			fb = llnormalpdf(new_value, 1.0) - llnormalpdf(old_value, 1.0);
// 			break;
// 		case EXPONENTIAL:
// 			new_value = random_exponential(1.0, RNG_ARGS);
// 			fb = lexponentialpdf(new_value, 1.0) - lexponentialpdf(old_value, 1.0);
// 			break;
// 		case UNIFORM:
// 			new_value = random_float(RNG_ARGS);
// 			fb = luniformpdf(new_value) - luniformpdf(old_value);
// 			break;			
// 	}
// 	h->constants[k] = new_value;
// 	return fb;	
// }


// a drift kernel
__device__ float resample_random_constant(hypothesis* h, RNG_DEF) {
	
	int k = random_int(MAX_CONSTANTS, RNG_ARGS);
	
	float old_value = h->constants[k];
	float new_value = old_value + random_normal(RNG_ARGS); // just drive a standard gaussian
	h->constants[k] = new_value;
	
	// since it's symmetric drift kernel proposal,
	return 0.0;	
}

// Take h and resample one of the constant *types*
// returning fb
// TODO: WE CAN MAKE THIS A local-GIBBS MOVE--compute the probability of the observed constant
// under each type and resample!!
__device__ float resample_random_constant_type(hypothesis* h, RNG_DEF) {
	int k = random_int(MAX_CONSTANTS, RNG_ARGS);
	h->constant_types[k] = random_int(__N_CONSTANT_TYPES, RNG_ARGS);
	return 0.0;
}

// Compute the prior on constats
// assuming constant types are generated uniformly
__device__ float compute_constants_prior(hypothesis* h) {
	
	float lp = 0.0;
	for(int k=0;k<MAX_CONSTANTS;k++) {
		float value = h->constants[k];
		
		switch(h->constant_types[k]) {
			case GAUSSIAN:    lp += lnormalpdf(value, 1.0); break;
			case LOGNORMAL:   lp += llnormalpdf(value, 1.0); break;
			case EXPONENTIAL: lp += lexponentialpdf(value, 1.0); break;
			case UNIFORM:     lp += luniformpdf(value); break;	
		}
	}
	
	return lp;
}

