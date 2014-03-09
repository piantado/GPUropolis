/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Hypothesis definitions and functions
 */

// the number of arguments to each function
#define nargs(x) NARGS[x]
#define stack_change(x) (1-nargs(x))

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Set program length and number of constants
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define MAX_PROGRAM_LENGTH 10
#define MAX_CONSTANTS 5

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The prior
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// the probabilities MUST sum to 1
const float P_X = -0.7985077;        // log(0.45);
const float P_CONSTANT = -0.7985077; // log(0.45);
const float P_F = -2.302585;         // log(0.1);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// what type is the program?
typedef char op_t;

// A struct for storing a hypothesis, which is a program and some accoutrements
// the check variables are there to make sure we don't overrun anything in any program editing
// main.cu has assertions to check they have CHECK_BIT value
// This also stores some ability for constants, but these are not used by all MH schemes
typedef struct hypothesis {
	float structure_prior;
	float constant_prior;
	float likelihood;
	float posterior;
	op_t   program[MAX_PROGRAM_LENGTH];
	float constants[MAX_CONSTANTS];
} hypothesis;

#define COPY_HYPOTHESIS(x,y) memcpy( (void*)(x), (void*)(y), sizeof(hypothesis));
#define COPY_CONSTANTS(x,y)  memcpy( (void*)((x)->constants), (void*)((y)->constants), sizeof(float)*MAX_CONSTANTS);


// A standard initialization that sets this thing up!
void initialize(hypothesis* h){
	h->posterior = -1.0/0.0;
	h->constant_prior = -1.0/0.0;
	h->structure_prior = -1.0/0.0;
	h->likelihood = 0.0;
	
	// zero the program
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++)
		h->program[i] = NOOP_; 
	
	// Set up our constants -- just random
	for(int i=0;i<MAX_CONSTANTS;i++) {
		h->constants[i] = 0.0;
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

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Bayes on hypotheses
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// These are used below but defined for real elsewhere
__device__ float f_output(float x, hypothesis* h, float* stack);

__device__ float compute_likelihood(int DLEN, datum* device_data, hypothesis* h, float* stack) {

	float ll = 0.0;
	for(int i=0;i<DLEN;i++){
		//__syncthreads(); // NOT a speed improvement
		
		float val = f_output( device_data[i].input, h, stack);
		
		// compute the difference between the output and what we see
		data_t d = device_data[i].output - val;
		
		// and use a gaussian likelihood
		ll += lnormalpdf(d, device_data[i].sd);
		
// 		printf(">> %f %f %f\n", val, d, ll);
	}
	
	
	// ensure valid
	if (!is_valid(ll)) ll = -1.0f/0.0f;
	
	return ll;
	
}


// Compute the prior on constats
// assuming constant types are generated uniformly
__device__ float compute_constants_prior(hypothesis* h) {
	
	float lp = 0.0;
	for(int k=0;k<MAX_CONSTANTS;k++) {
		lp += lnormalpdf(h->constants[k], 1.0);
	}
	
	return lp;
}

// Compute the prior on constats
// assuming constant types are generated uniformly
float compute_structure_prior(hypothesis* h) { // CANNOT be run on device; must use NARGS instead of hNARGS
	
	float lp = 0.0;
	for(int k=0;k<MAX_PROGRAM_LENGTH;k++) {
		op_t pi = h->program[k];
		
		if(pi == NOOP_) continue;
		
		if(hNARGS[pi]==0) {
			if(pi == X_) { lp += P_X; }
			else         {
				       lp += P_CONSTANT - log(3.); 
				       assert(pi == CONSTANT_ || pi == ONE_ || pi == ZERO_);
			}
		} else {
			lp += P_F;
		}
	}
	
	return lp;
}



__device__ float update_posterior(int DLEN, datum* device_data, hypothesis* h, float* stack) {
	// Compute all of the posterior stats and save them

	h->likelihood = compute_likelihood(DLEN, device_data, h, stack);
	h->constant_prior = compute_constants_prior(h);
	
	h->posterior = h->constant_prior + h->structure_prior + h->likelihood;
// 	printf("%f %f %f \n", h->constant_prior, h->structure_prior, h->likelihood);
	
	return h->posterior;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Manipulate constants
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__device__ float resample_random_constant(hypothesis* h, RNG_DEF) {
	// Resample, and then return f-b (which is 0)
	
	float sdf = random_float(RNG_ARGS);
	
	// Mix together different SDs, for differnet scales
	float               SD =  0.1; 
	if(sdf < 0.10)      SD =  1.0;
	else if(sdf < 0.15) SD = 10.0;
	
	// and change a bernoulli number of them
	for(int k=0;k<MAX_CONSTANTS;k++) {
		if(random_float(RNG_ARGS) < 0.2) {
			h->constants[k] += SD * random_normal(RNG_ARGS);
		}
	}
	
	// since it's symmetric drift kernel proposal:
	return 0.0;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Enumeration of hypotheses
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void enumerate_all_programs_rec( hypothesis* cur, int pos, int nopen, int maxdepth, int (*callback)(hypothesis*) ) {
	// Call callback on each hypothesis generated, via enumeration
	if(pos < MAX_PROGRAM_LENGTH-maxdepth) return;
		
	for(int oi=NOOP_+1;oi<NUM_OPS;oi++) { // START right after NOOP_, which must be first!
		cur->program[pos] = oi;
		int nowopen = nopen + (1-hNARGS[oi]); // CANNOT Use stack_change since that doesn't access hNARGS

		if(nowopen == 0) {
			
			for(int i=0;i<MAX_CONSTANTS;i++) {
				cur->constants[i] = 0.0; // just to start random_normal(RNG_ARGS);
			}

			int keepgoing = callback(cur);
			if(!keepgoing) break; // exit early
		}
		else {
			enumerate_all_programs_rec(cur, pos-1, nowopen, maxdepth, callback);
			cur->program[pos-1] = NOOP_; // undo this when we're done
		}
		
	}
}
void enumerate_all_programs(int maxn, int maxdepth, int(*callback)(hypothesis*) ) {
	assert(maxdepth <= MAX_PROGRAM_LENGTH);
	
	hypothesis* cur = new hypothesis();
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) cur->program[i] = NOOP_; // initialize to zeros
	
	enumerate_all_programs_rec(cur, MAX_PROGRAM_LENGTH-1, -1, maxdepth, callback);
}


