


IN THIS ONE I HAD TRIED TO FIT CONSTANTS, BUT IT DIDN'T MIX (and im not sure i got detailed balance right), probably due to the fact that arith ops do weird things to constants, and invert any prefs we have.

So the next version has no constants, only 1, and the prior on constants is derived through arith. 









/*
 * 
 * 
 * 
 * -- TOOD : Make it so that it doesn't re-compute log of prior each go-roun
 * -- FIX NANs in posteriors sorting -- they mess everything up
 * 
 * -- tHESE CONSTANTS SEEM TO HAVE A REALLY BIG INFLUENCE ON WHAT POSTERIORS END UP LOOKING LIKE.
 * -- To run on CPU: http://code.google.com/p/gpuocelot/
 * -- If we want the prior not to depend on the scale, we must build in some logarithmic steps, and treat them equally
 * -- If we want the likelihood not to depend on the scale, we may want a ratio likelihood... but the downside of that is that its not shift-invariant (so if you shift near zero, things go to hell)
 * - Once you add samples, you can resample for a number of steps and output every number of steps
 * - REPLACE &(x[0]) with &x, I think will work!
 * 
 * TODO:
 * 	- Bundle together a hypothesis so its easy to use, and you can get rid of the posterior return
 * 	- Create a local copy of the input and output arrays
 * 	- Can run at a variety of temperatures
 * 	- We can examine the "health" of the chain by seeing how many times we find the "top" one
 * 	-- Huh, probably by particle re-sampling, we can explore the really good regions of the space even better since that moves everyone to one of those regions
 * 
 * 	- The constant stack should be either as large as program (to handle that) or handled better in computing the output
 * 
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>

#include <vector>
using namespace std;

// this is our arithmetic if/then macro
#define ifthen(x,y,z) ((y)*(x) + (z)*(1-(x)))

// Check if something is not a nan or inf
#define is_valid(x) (!(isnan(x) || isinf(x)))

const int N = 1000;  // oHw many chains?
const int BLOCK_SIZE = 512;
const int N_BLOCKS = N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1);
	
// int DLEN; // how long is the data?
// float* INPUT; // = { 1.0, 2.0, 3.0, 4.0,5.0, 6.0, 7.0 };
// float* OUTPUT;// = { 1.0000000, 0.7071068, 0.5773503, 0.5000000, 0.4472136, 0.4082483, 0.3779645}; // x**-0.5
// // __constant__ const float OUTPUT[DLEN] = { 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0 };
// __constant__ const float OUTPUT[DLEN] = { 0.000000000, -0.6065307, -1.4330626, -2.3364023, -3.2749230, -4.2324086, -5.2012674 }; // (1-x) / exp(1/x)

// The file that we read input from
char* INPUT_FILE = "datasets/hubble.txt"; //"datasets/squared.txt";
// __constant__ const float OUTPUT[DLEN] = { 0.0, 0.4041569,0.9990636,1.5557204,2.0408044,2.4605594,2.8264565}; // log(1+x*(x-1)) / (exp(2/x))

const float PRIOR_TEMPERATURE = 1.0;
const int MCMC_ITERATIONS = 100000;

const float LL_SIGMA = 1.0;

const float RESAMPLE_P = 0.15; // when we propose, with what probability do we change each program element?

const int MAX_PROGRAM_LENGTH = 30; 
const int MAX_CONSTANT_COUNT = 3; // how many constants do we allow?

const int OUTER_BLOCKS = 1;
const int TOP_RECOPY = N; // resample flat among these top hypotheseses


__constant__ const float PIf = 3.141592653589;

// const int OUTER_SAMPLES = 10; // how many times do we sample from each chain?

// CONSTANTS, 2-ARY, 1-ARY
// NUM_OPS stores the number of opsn
// TODO: IF YOU CHANGE THESE, MYOU MSUT CHANGE THE CONDITION BELOW FOR DECODING WHAT HAPPENS TO THE TOP OF STACK
enum OPS                              { X, POPCONS, ADD, SUB, MUL, DIV, POW,     LOG, EXP, SIN, COS, TAN, NUM_OPS };
__constant__ int NARGS[NUM_OPS]     = { 0,   0,     2,   2,   2,   2,   2,        1,  1,    1,   1,   1 }; // how many args for each op?
__constant__ float prior[NUM_OPS]   = { 0.25, 0.25,0.05, 0.05,0.05,0.05,0.05,   0.05,0.05,0.05,0.05,0.05};

__constant__ float NUM_OPSf = float(NUM_OPS);

enum HELD_OUT { NONE, HALF, ODD}; // which data elements do we hold out?

// how much does each op change the stack?
#define stack_change(x) (1-NARGS[x]) 

// A struct for storing a hypothesis, which is a program and some accoutrements
typedef struct {
	float prior;
	float likelihood;
	float posterior;
	float temperature;
	float llsigma; 
	float constant_scale; // the scale for the prior on constants
	int   program[MAX_PROGRAM_LENGTH];
	float constants[MAX_CONSTANT_COUNT]; 
} hypothesis;

// so we qsort so best is LAST
int hypothesis_posterior_compare(const void* a, const void* b) { return ((hypothesis*)a)->posterior > ((hypothesis*)b)->posterior; }
int hypothesis_structurally_identical( hypothesis* a, hypothesis* b) {
	// check if two hypotheses are equal in terms of the program (ignoring constants and other params)
	for(int i=MAX_PROGRAM_LENGTH;i>=0;i--) {
		if(a->program[i] != b->program[i]) return 0;
	}
	return 1;
}
// swap two pointers conditionally, without branching
__device__ void ifthenswap(int Q, void** x, void** y, void** tmp) {
	int i = int(Q>0);
	tmp[0] = *x;
	tmp[1] = *y;
	void* t1 = tmp[i];
	void* t2 = tmp[1-i];
	(*x) = t1;
	(*y) = t2;
}

// a random number 0..(n-1), using the stored locations for x,y,z,q
__device__ int random_int(int n, int& x, int& y, int& z, int& w) {
	int t;
 
	t = x ^ ( x << 11);
	x = y; y = z; z = w;
	w = w ^ (w >> 19) ^ (t ^ (t >> 8));
	
	return (w%n);
}

__device__ float random_float(int& x, int& y, int& z, int& w) {
	return float(random_int(1000000, x,y,z,w)) / 1000000.0;
}

// Choose an op from the prior distribution
// NOTE: assumes prior on expansions is normalized
__device__ int random_op(int& x, int& y, int& z, int& w) {
	float f = random_float(x,y,z,w);
	int ret = 0;
	int notdone = 1;
	for(int i=0;i<NUM_OPS;i++) {
		f = f-prior[i];
		ret = ret + i*(f<0.0)*notdone;
		notdone = (f>=0.0);
	}
	return ret;
}

__device__ float f_output(float x, hypothesis* h, float* registers, float* stack) {

	// first zero the stack
	for(int i=0;i<2*MAX_PROGRAM_LENGTH;i++) stack[i] = 0.0; 
	
	registers[X]    = x;
	
	int top = MAX_PROGRAM_LENGTH; //  We start in the middle of the stack
	int ctop = 0x0; // the top of the constant stack
	for(int p=0;p<MAX_PROGRAM_LENGTH;p++) { // program pointer
		int op = h->program[p];
		
		// update the virtual registers
		registers[ADD]  = stack[top] + stack[top-1];
		registers[SUB]  = stack[top] - stack[top-1];
		registers[MUL]  = stack[top] * stack[top-1];
		registers[DIV]  = stack[top] / stack[top-1];
		registers[POW]  = pow(stack[top], stack[top-1]);
		registers[LOG]  = log(stack[top]);
		registers[EXP]  = exp(stack[top]);
		registers[SIN]  = sin(stack[top]);
		registers[COS]  = cos(stack[top]);
		registers[TAN]  = tan(stack[top]);
		
		registers[POPCONS] = h->constants[ctop];
		
		// the *(op<NUM_OPS&&op>=0) here makes us handle accidentally huge numbers, treating them as 0s
		// what elements changes? If we have a constant, we push; a 1-ary and we replace; a 2-ary and we gobble one
		top += stack_change(op)*(op<NUM_OPS&&op>=0); // so 0 args push, 1 args make no change, 2args eat the top
		stack[top] = registers[op*(op<NUM_OPS&&op>=0)];
		
		// and update the constant stack if we need to
		ctop += (op==POPCONS);
	}
	
	return stack[top];
}

__device__ float lnormalpdf( float x, float s ){
	return -(x*x)/(2.0*s*s) - 0.5 * log(2.0*PIf*s*s);
}

// Generate a 0-mean random deviate with mean 0, sd 1
// This uses Box-muller so we don't need branching
__device__ float random_normal(int& x, int& y, int& z, int& w) {
	float u = random_float(x,y,z,w);
	float v = random_float(x,y,z,w);
	return sqrt(-2.0*log(u)) * sin(2*PIf*v);
}

__device__ float random_lnormal(float u, float s, int& x, int& y, int& z, int& w) {
	return exp(u+s*random_normal(x,y,z,w));
}

// Log normal
__device__ float llnormalpdf(float x, float s) {
	return lnormalpdf( log(x), s) - log(x);
}

// Puts a random closed expression into buf, pushed to the rhs
// and returns the start position of the expression. 
// NOTE: Due to size constraints, this may fail, which is okay, it will 
// just create a garbage proposal
__device__ int random_closed_expression(int* buf, int& x, int& y, int& z, int& w) {
	
	int nopen = -1; // to begin, we are looking for 1 arg to the left
	int notdone = 1;
	int len = 0;
	for(int i=MAX_PROGRAM_LENGTH-1;i>=0;i--) {
		int newop = random_op(x,y,z,w); // random_int(NUM_OPS,x,y,z,w);
		buf[i] = notdone * newop;
		
		nopen += notdone * stack_change(newop); //(1-NARGS[newop]);
		len += notdone;

		notdone = notdone && (nopen != 0); // so when set done=0 when nopen=0 -- when we've reached the bottom!
	}
	
	return (MAX_PROGRAM_LENGTH-len);	
}

// Starting at i in buf, go forwards and return the position of 
// the matching paren (corresponding to an expression)
__device__ int find_close_backwards(int* buf, int pos) {
	
	int nopen = -1;
	int notdone = 0;
	int ret = pos+1;
	for(int i=MAX_PROGRAM_LENGTH-1;i>=0;i--) {
		notdone =  (i==pos) || (notdone && nopen != 0);
		nopen  +=  (i<=pos && notdone) * stack_change(buf[i]);
		ret    -=  (i<=pos && notdone); 
	}
	return ret;	
}

// set ar[start_ar...(start_ar+len_ar)] = x[MAX_PROGRAM_LENGTH-len_x:MAX_PROGRAM_LENGTH-1];
// nonincludsive of end_ar, but inclusive of start_ar. inclusive of start_x
// NOTE: WE can get garbage on the left of our string if we insert soemthing in 0th (rightmost) position
__device__ void special_splice(int* ar, int start_ar, int end_ar, int* x, int start_x, int* dest) {
	
	// correct for the mis-alignment of x and the gap of what its replacing
	int shift = (MAX_PROGRAM_LENGTH - start_x) - (end_ar+1-start_ar); 
	
	int xi = start_x;
	for(int ari=0;ari<MAX_PROGRAM_LENGTH;ari++) {
		int in_splice_region = (ari>=start_ar-shift) && (ari<=end_ar);
		int in_final_region = (ari > end_ar);
		
		// wrap this for in case it goes off the end
		int ar_ari_shift = ifthen( (ari+shift < 0) || (ari+shift >= MAX_PROGRAM_LENGTH), 0, ar[ari+shift]); 
// 		dest[ari] = ifthen(in_splice_region, 2, ifthen(in_final_region, 3, 1) );
		
		dest[ari] = ifthen(in_splice_region, x[xi], ifthen(in_final_region, ar[ari], ar_ari_shift) );
		
		xi += in_splice_region; // when in the splice region, increment by 1
	}
	
}

// starting at the RHS, go until we find the "Effective" program length (ignoring things that are not used)
__device__ int find_program_length(int* program) {
	return MAX_PROGRAM_LENGTH-find_close_backwards(program, MAX_PROGRAM_LENGTH-1);
}




__device__ float compute_likelihood(int DLEN, float* device_INPUT, float* device_OUTPUT, hypothesis* h, float* registers, float* stack) {

	float ll = 0.0;
	for(int i=0;i<DLEN;i++){
		// compute the difference between the output and what we see
		float d = device_OUTPUT[i] - f_output( device_INPUT[i], h, registers, stack);
		
		// and use a gaussian likelihood
		ll += lnormalpdf(d, h->llsigma);
	}
	return ll;
}

// the prior just on the program part
__device__ float compute_program_prior(hypothesis* h) {
	float lprior = 0.0;
	int close = find_program_length(h->program);
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
		lprior     += (i>=close)*log(prior[h->program[i]]); // prior for this op
	}
	return lprior;
}

__device__ int count_constants(hypothesis* h) {
	int close = find_program_length(h->program);
	int constcount = 0;
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++)
		constcount += (i>=close)*(h->program[i] == POPCONS);
	return constcount;
}

// Compute the program's prior, 2^-length, but note that the program must be pushed to the "right" in the array,
// so we can do this by counting the number of leading zeros (TODO: WHICH ISN'T QUITE RIGHT)
__device__ float compute_prior(hypothesis* h) {
	
	float lprior = compute_program_prior(h); // the prior just on the program, and ad the rest
	int constcount = count_constants(h);
	
	// prior on each constant
	for(int i=0;i<MAX_CONSTANT_COUNT;i++) {
		
		// a normal prior makes you want to push everything to zero. Which gives you tiny constants; not intuitively right
// 		lprior += lnormalpdf( h->constants[i], h->constant_scale) * (i<constcount);
		lprior += llnormalpdf(h->constants[i], h->constant_scale) ; //* (i<constcount); // TODO: MAYBE DON'T INLCUDE THIS IF WE IGNORE ADDITIONAL CONSTANTS BELOW
	}
	
	// and the prior on llsigma -- Put an exponential prior on these
	lprior += -h->llsigma; 
	lprior += -h->constant_scale; 
	
	return lprior / float(PRIOR_TEMPERATURE);
}

__device__ void compute_posterior(int DLEN, float* device_INPUT, float* device_OUTPUT, hypothesis* h, float* registers, float* stack) {
	h->prior      = compute_prior(h);
	h->likelihood = compute_likelihood(DLEN, device_INPUT, device_OUTPUT, h, registers, stack);
	h->posterior = h->prior + h->likelihood;
}

// Kernel that executes on the CUDA device
// initialize_sample here will make us resample if 1; else we use out_hypothesis as-is and propose form that
__global__ void device_run(int N, int DLEN, float* device_INPUT, float* device_OUTPUT, hypothesis* out_hypotheses, int seed, int initialize_sample)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Set up for the RNG
	int rx = idx*seed; // 123456789; // MUST be 32 bit //!!                    <- TODO: HERE WE UNDID THE RANDOM SEED
	int ry = 362436069;
	int rz = 521288629;
	int rw = 88675123;
	
	float stack[2*MAX_PROGRAM_LENGTH]; // Stack is twice as big as program, and populated with 0s so we can eval, e.g., a whole prog of ADDs
	float registers[NUM_OPS];
	
	hypothesis current_;  hypothesis* current = &current_;
	hypothesis proposal_; hypothesis* proposal = &proposal_;
	
	int program_buf_[MAX_PROGRAM_LENGTH]; // needed below as a buffer program
	int* program_buf = &(program_buf_[0]);
	
	void* swaptmp[2]; // used for swapping without arithmetic
		
	if(initialize_sample) { // randomly initialize if we should
		random_closed_expression(current->program,  rx,ry,rz,rw);
		for(int i=0;i<MAX_CONSTANT_COUNT;i++)
			current->constants[i] = 1.0;
		
// 			current->constants[i] = random_lnormal(0.0, proposal->constant_scale, rx,ry,rz,rw);
// 		current->llsigma = LL_SIGMA;
// 		current->constant_scale = 1.0;
		current->llsigma = 1.0;
		current->constant_scale = 1.0;
	}
	else { // Else copy over
		memcpy((void*)current, out_hypotheses + idx, sizeof(hypothesis));
	}
		
	compute_posterior(DLEN, device_INPUT, device_OUTPUT, current, &(registers[0]), &(stack[0]));
		
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<MCMC_ITERATIONS;mcmci++) {
		
		
		for(int pp=0;pp<=4;pp++) { // for each kind of proposal

			float fb = 0.0;  // the forward-backward probability
			int len = find_program_length(current->program); // TODO: CHECK OFF BY !
			
			// for now, JUST do MCMC ON THE constants:			
			fb = 0.0; // these proposals should be symmetric...
			for(int i=0;i<MAX_CONSTANT_COUNT;i++) {
				float c = random_lnormal(0.0, proposal->constant_scale, rx,ry,rz,rw);
				fb += lnormalpdf(c, proposal->constant_scale) - lnormalpdf(proposal->constants[i], proposal->constant_scale);
				proposal->constants[i] = c;
			}
			proposal->llsigma = current->llsigma;
			proposal->constant_scale = current->constant_scale;
			
			
			
			
			if(pp==0) { // full regeneration
// 				fb = 0.0;
// 				proposal->llsigma = current->llsigma;
// 				proposal->constant_scale = current->constant_scale;
// 				
// 				random_closed_expression(proposal->program, rx, ry, rz, rw);
// 				for(int i=0;i<MAX_CONSTANT_COUNT;i++) {
// 					proposal->constants[i] =  random_lnormal(0.0, proposal->constant_scale, rx,ry,rz,rw);
// 					
// 					//!! TODO: CHECK THAT WE SHOULDN'T IGNORE THINGS BEYOND HTE NUM OF CONST WE HAVE
// 					fb += llnormalpdf(proposal->constants[i], proposal->constant_scale) - llnormalpdf(current->constants[i], current->constant_scale);
// 				}
// 				
// 				fb += compute_program_prior(proposal) - compute_program_prior(current);
			}
			else if(pp==1) { // normal tree regen
				
// 				// first find a proposal location:
// 				int end_ar = (MAX_PROGRAM_LENGTH-1) - random_int(len-1, rx,ry,rz,rw); // the position of what we replace
// 				int start_ar = find_close_backwards(current->program, end_ar);
// 				
// 				// generate a novel tree in program_buf
// 				int start_x = random_closed_expression(program_buf, rx,ry,rz,rw);
// 				special_splice(current->program, start_ar, end_ar, program_buf, start_x, proposal->program); // And insert this where we wanted:
// 				
// 				float proposed_len = (float) find_program_length(proposal->program);
// 				float current_len = (float) find_program_length(current->program);
// 				
// 				// forward-back probability
// 				fb = (-log(current_len)+compute_program_prior(proposal) ) - (-log(proposed_len)+compute_program_prior(current));
			}
			else if(pp==2) {
// 				// propose to constant_scale
				float c = random_lnormal(0.0, 1.0, rx,ry,rz,rw);
				fb = llnormalpdf(c, 1.0) - llnormalpdf(proposal->constant_scale, 1.0);
				proposal->constant_scale = c;
			}
			else if(pp==3) {
// 				float c = random_lnormal(0.0, 1.0, rx,ry,rz,rw);
// 				fb = llnormalpdf(c, 1.0) - llnormalpdf(proposal->llsigma, 1.0);
// 				proposal->llsigma = c;
			}
			else if(pp == 4) {
// 				fb = 0.0; // these proposals should be symmetric...
// 				for(int i=0;i<MAX_CONSTANT_COUNT;i++) {
// 					float c = random_lnormal(0.0, proposal->constant_scale, rx,ry,rz,rw);
// 					fb += lnormalpdf(c, proposal->constant_scale) - lnormalpdf(proposal->constants[i], proposal->constant_scale);
// 					proposal->constants[i] = c;
// 					
// 					// sinc eour prior is lognormal, we'll make lognormal proposals
// // 					proposal->constants[i] = random_normal(rx,ry,rz,rw) * proposal->constant_scale;
// 				}
			}
// 			else if(pp==5) {
// 				
// 				int pos = (MAX_PROGRAM_LENGTH-1) - random_int(len-1, rx,ry,rz,rw);
// 				
// 				// random insert into a program
// 				for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
// 					proposal->program[i] = current->program[i+1]*(i<pos) + random_int(NUM_OPS,rx,ry,rz,rw)*(i==pos) + current->program[i]*(i>pos);
// 				}
// 				
// 				fb = (-log(float(len)) - log(NUM_OPSf) ) - (-log(float(len+1)) );
// 				
// 			}
// 			else if(pp==6) {
// 				int pos = (MAX_PROGRAM_LENGTH-1) - random_int(len-1, rx,ry,rz,rw);
// 				
// 				// random delete from a program
// 				proposal->program[0] = 0x0;				
// 				for(int i=1;i<MAX_PROGRAM_LENGTH;i++) {
// 					proposal->program[i] = current->program[i-1]*(i<=pos) + current->program[i]*(i>pos);
// 				}
// 				
// 				fb = (-log(float(len)) ) - (-log(float(len-1)) - log(NUM_OPSf) );
// 				
// 			}
// 			else if(pp==7) {
				
// 				// Simple stupid proposal that just mutate at random:	 
// 				for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
// 					int r = random_float(rx,ry,rz,rw) < RESAMPLE_P;
// 					proposal->program[i] = ifthen(r, random_int(NUM_OPS,rx,ry,rz,rw), current->program[i]);
// 				}
// 			}
			
			// compute the posterior for the proposal
			compute_posterior(DLEN, device_INPUT, device_OUTPUT, proposal, &(registers[0]), &(stack[0]));
			
			// compute whether not we accept the proposal, while rejecting infs and nans
			int swap = (random_float(rx,ry,rz,rw) < exp(proposal->posterior - current->posterior - fb) && is_valid(proposal->posterior)) || (! is_valid(current->posterior));
			
			// Use a trick to swap pointers without branching 
			ifthenswap(swap, (void**)&current, (void**)&proposal, swaptmp);
			
		} // end for each proposal kind
	}
	
	if (idx<N) {
		memcpy(out_hypotheses + idx, (void*)current, sizeof(hypothesis));
	}
}
 
 
 
//------------
// For easier displays
const int MAX_OP_LENGTH = 256; // how much does each string add at most?
char SS[MAX_PROGRAM_LENGTH*2][MAX_OP_LENGTH*MAX_PROGRAM_LENGTH]; 
void print_program_as_expression(hypothesis* h) {
	
	char buf[MAX_PROGRAM_LENGTH*MAX_PROGRAM_LENGTH];
	
	int top = MAX_PROGRAM_LENGTH; // top of the stack
	int ctop = 0x0; // the top of the ocnstant stack 
	
	// re-initialize our buffer
	for(int r=0;r<MAX_PROGRAM_LENGTH*2;r++) strcpy(SS[r], "0"); // since everything initializes to 0
	
	for(int p=0;p<MAX_PROGRAM_LENGTH;p++) {
		int op = h->program[p];
		
		switch(op) {
			case POPCONS:
				top += 1;
				sprintf(SS[top], "%f", h->constants[ctop]);
				ctop++;
				break;
// 			case ZERO: 
// 				top += 1;
// 				strcpy(SS[top], "0");
// 				break;
// 			case ONE: 
// 				top += 1;
// 				strcpy(SS[top], "1");
// 				break;
			case X:
				top += 1;
				strcpy(SS[top], "x");
				break;
// 			case PI:
// 				top += 1;
// 				strcpy(SS[top], "PI");
// 				break;
// 			case E:
// 				top += 1;
// 				strcpy(SS[top], "E");
// 				break;
			case ADD:
				strcpy(buf, "add(");
				strcat(buf, SS[top]);
				strcat(buf, ",");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case SUB:
				strcpy(buf, "sub(");
				strcat(buf, SS[top]);
				strcat(buf, ",");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case MUL:
				strcpy(buf, "mul(");
				strcat(buf, SS[top]);
				strcat(buf, ",");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case DIV:
				strcpy(buf, "div(");
				strcat(buf, SS[top]);
				strcat(buf, ",");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
			case POW:
				strcpy(buf, "pow(");
				strcat(buf, SS[top]);
				strcat(buf, ",");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;	
				
			case LOG:
				strcpy(buf, "log(");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
				
			case EXP:
				strcpy(buf, "exp(");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
			case SIN:
				strcpy(buf, "sin(");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
			case COS:
				strcpy(buf, "cos(");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
			case TAN:
				strcpy(buf, "tan(");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
				
		}
	}
	
	printf("%s", SS[top]);
}
 
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
// main routine that executes on the host
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
int main(void)
{	
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	// The data:
	FILE* fp = fopen(INPUT_FILE, "r");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << INPUT_FILE <<"\n"; return 1;}
	
	vector<float> inx; float x;
	vector<float> iny; float y;
	char* line = NULL; size_t len=0;
	while( getline(&line, &len, fp) != -1) {
		if (sscanf(line, "%f\t%f\n", &x, &y) == 2) { 
			inx.push_back(x);
			iny.push_back(y);
		}
	}
	
	assert(inx.size() == iny.size());
	const int DLEN = inx.size();
	const size_t DATA_BYTE_LEN = DLEN*sizeof(float);
	
	size_t out_hypotheses_size = N * sizeof(hypothesis);
	hypothesis* host_out_hypotheses = new hypothesis[N]; 
	hypothesis* device_out_hypotheses; cudaMalloc((void **) &device_out_hypotheses, out_hypotheses_size); // device allocate
	 
	
	// copy the read input to the GPU
	float* device_INPUT;
	cudaMalloc((void **) &device_INPUT, DATA_BYTE_LEN);
	float* host_INPUT = (float*) malloc(DATA_BYTE_LEN);
	for(int i=0;i<DLEN;i++) host_INPUT[i] = inx[i];
	cudaMemcpy(device_INPUT, host_INPUT, DATA_BYTE_LEN, cudaMemcpyHostToDevice);
	
	float* device_OUTPUT;
	cudaMalloc((void **) &device_OUTPUT, DATA_BYTE_LEN);
	float* host_OUTPUT = (float*) malloc(DATA_BYTE_LEN);
	for(int i=0;i<DLEN;i++) host_OUTPUT[i] = iny[i];
	cudaMemcpy(device_OUTPUT, host_OUTPUT, DATA_BYTE_LEN, cudaMemcpyHostToDevice);
	
	// Do calculation on device:
	srand(time(NULL));
	int seed = rand();
	
	
	for(int outer=0;outer<OUTER_BLOCKS;outer++) {
		
		device_run<<<N_BLOCKS,BLOCK_SIZE>>>(N, DLEN, device_INPUT, device_OUTPUT, device_out_hypotheses, seed+outer, outer==0);
		
		// Retrieve result from device and store it in host array
		cudaMemcpy(host_out_hypotheses, device_out_hypotheses, out_hypotheses_size, cudaMemcpyDeviceToHost);
		
		// sort the samples by probability:
		qsort( (void*)host_out_hypotheses, N, sizeof(hypothesis), hypothesis_posterior_compare);
		
		//Print results
		for(int n=0; n<N; n++){
			printf("%d\t%d\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t", outer, n, host_out_hypotheses[n].prior+host_out_hypotheses[n].likelihood,  host_out_hypotheses[n].prior, host_out_hypotheses[n].likelihood, host_out_hypotheses[n].llsigma, host_out_hypotheses[n].constant_scale);
			
			printf("\"");
			for(int i=0;i<MAX_PROGRAM_LENGTH;i++) 
				printf("%d ", host_out_hypotheses[n].program[i]);
			printf("\"\t");
					
			for(int i=0;i<MAX_CONSTANT_COUNT;i++) 
				printf("%f\t", host_out_hypotheses[n].constants[i]);
			
			printf("\"");
			print_program_as_expression( &(host_out_hypotheses[n]) );
			printf("\"\n");
		}
		fflush(stdout);
		
		
		// Now go through and find the top TOP_RECOPY structurally unique hypotheses
		// and put them at the end
		int j = N-1;
		for(int i=N;i>=0 && j>=N-TOP_RECOPY;i--) {
			int keep = 1;
			
			for(int chk=j+1;chk<N;chk++){ // check to see if this is identical to anything previous that we've kept
				if(hypothesis_structurally_identical(  &(host_out_hypotheses[i]),  &(host_out_hypotheses[chk] )) ) {
					keep = 0;
					break;
				}
			}
			
			if(keep) {
				memcpy( (void*)&(host_out_hypotheses[j]), (void*)&(host_out_hypotheses[i]), sizeof(hypothesis)); 
				j--; 
			}
		}
		// We will take the top N and recopy along in the host
		
		
		for(int i=N-1;i>=N-TOP_RECOPY;i--) { // copy the best, which start from the end
			for(int j=i-TOP_RECOPY;j>=0; j -= TOP_RECOPY) // and go backwards
				memcpy( (void*)&(host_out_hypotheses[j]), (void*)&(host_out_hypotheses[i]), sizeof(hypothesis)); 
			
		}
		
		cudaMemcpy(device_out_hypotheses, host_out_hypotheses, out_hypotheses_size, cudaMemcpyHostToDevice);
		
	}

	// Cleanup
	delete[] host_out_hypotheses;
	delete[] host_OUTPUT;
	delete[] host_INPUT;
	
	cudaFree(device_OUTPUT);
	cudaFree(device_INPUT);
	cudaFree(device_out_hypotheses);
}