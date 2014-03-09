/*
 * 
 * 
 * 
 * 
 *  - DEfine a "program type" (currently int) for programs, and then we can test out short and other things
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 * sORT IS NOT WORKING RIGHT
 * FOr SOME REASON WE GET THESE CRAZY HIGH PRIOR+LIKELIHOODS AND HTE SUMS ARE NONSESE
 * 
 * ANd if w efix the prior or likleihod, we get garbage, with all 0s in the proposal etc. 
 * 
 * 
 * this appears to be a problem using the tree regeneration
 * HMM SOMEHOW WE GET nans.. WTF
 * 
 * 
 * 
 * 
 * 
 * 
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



const int N = 5000;  // Hw many chains?
const int BLOCK_SIZE = 1000;
const int N_BLOCKS = N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1);

char* INPUT_FILE = "datasets/polynomial2.txt"; 

const int MAX_PROGRAM_LENGTH = 75; 

const float PRIOR_TEMPERATURE = 1.0; // WARNING: If you make this >=1, our chain, proposing from the prior, may wander into weird long equiv strings
const int MCMC_ITERATIONS = 1000000; //500000;
const float RESAMPLE_P = 0.15; // when we propose, with what probability do we change each program element?

const float LL_SIGMA = 1.0;

const int OUTER_BLOCKS = 1;
const int TOP_RECOPY = 100; // resample flat among these top hypotheseses
const float STACK_EPSILON = 1e-5; // things less than this in absolute value are collapsed to zero, to prevent things like log(1-1)

#include "src/misc.cu"
#include "src/primitives.cu"
#include "src/hypothesis.cu"
#include "src/programs.cu"

__device__ float f_output(float x, hypothesis* h, float* registers, float* stack) {

	// first zero the stack
	for(int i=0;i<2*MAX_PROGRAM_LENGTH;i++) stack[i] = 0.0; 
	
	registers[X]   = x;
	registers[ONE] = 1.0;
	
// 	int garbageflag = 0; // this is set to 1 if ANY subcomputation results in NAN. This way, we don't have CUDA craziness like passings nans to pow 
	                     // if this is 1, we return nan
	
	int top = MAX_PROGRAM_LENGTH; //  We start in the middle of the stack
	for(int p=0;p<MAX_PROGRAM_LENGTH;p++) { // program pointer
		int op = h->program[p];
		
		// update the virtual registers
		registers[ADD]  = stack[top] + stack[top-1];
		registers[SUB]  = stack[top] - stack[top-1];
		registers[MUL]  = stack[top] * stack[top-1];
		registers[DIV]  = stack[top] / stack[top-1];
		registers[NEG]  = -stack[top];
		registers[POW]  = pow(stack[top], stack[top-1]);
		registers[LOG]  = log(stack[top]);
		registers[EXP]  = exp(stack[top]);
		registers[SIN]  = sin(stack[top]);
		registers[ASIN] = asin(stack[top]);
		
		// the *(op<NUM_OPS&&op>=0) here makes us handle accidentally huge numbers, treating them as 0s
		// what elements changes? If we have a constant, we push; a 1-ary and we replace; a 2-ary and we gobble one
		top += stack_change(op)*(op<NUM_OPS&&op>=0); // so 0 args push, 1 args make no change, 2args eat the top
		
		float newtop = registers[op*(op<NUM_OPS&&op>=0)];
		
		// we correct floating point error towards zero to prevent things like log(1-x/x) have a non-inf value
		// It's actually a pretty amazing fact that this thing will exploit this property of floating point numbers!!
// 		stack[top] = float( abs(newtop) > STACK_EPSILON ) * newtop;
		stack[top] = newtop;
		
// 		garbageflag = garbageflag || ( isnan(newtop) );
	}
	
// 	return stack[top]*(garbageflag==1); //!! THIS WILL MAKE 0 FOR ALL NAN OUTPUT
	return stack[top];
}

__device__ float mul(float x, float y) { return x*y; }
__device__ float div(float x, float y) { return x/y; }
__device__ float add(float x, float y) { return x+y; }
__device__ float sub(float x, float y) { return x-y; }

// Kernel that executes on the CUDA device
// initialize_sample here will make us resample if 1; else we use out_hypothesis as-is and propose form that
__global__ void device_run(int N, int DLEN, float* device_INPUT, float* device_OUTPUT, hypothesis* out_hypotheses, int seed, int initialize_sample)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Set up for the RNG
	int rx = idx*seed; // 123456789; // MUST be 32 bit
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
		current->llsigma = LL_SIGMA;
	}
	else { // Else copy over
		memcpy((void*)current, out_hypotheses + idx, sizeof(hypothesis));
	}
		
	compute_posterior(DLEN, device_INPUT, device_OUTPUT, current, registers, stack);
		
	// Now main MCMC iterations	
	for(int mcmci=0;mcmci<MCMC_ITERATIONS;mcmci++) {
		
		
		for(int pp=0;pp<=2;pp++) { // for each kind of proposal

			float fb = 0.0;  // the forward-backward probability
			int len = find_program_length(current->program); // TODO: CHECK OFF BY !
			
			if(pp==0) { // full regeneration
				fb = 0.0;
				proposal->llsigma = current->llsigma;
				random_closed_expression(proposal->program, rx, ry, rz, rw);
				fb += compute_program_prior(proposal) - compute_program_prior(current);
			}
			else if(pp==1) {
				float c = random_lnormal(0.0, 1.0, rx,ry,rz,rw);
				fb = llnormalpdf(c, 1.0) - llnormalpdf(proposal->llsigma, 1.0);
				proposal->llsigma = c;
			}
			else if(pp==2) { // normal tree regen
				
				// first find a proposal location:
				int end_ar = (MAX_PROGRAM_LENGTH-1) - random_int(len-1, rx,ry,rz,rw); // the position of what we replace
				int start_ar = find_close_backwards(current->program, end_ar);
				
				// generate a novel tree in program_buf
				int start_x = random_closed_expression(program_buf, rx,ry,rz,rw);
				special_splice(current->program, start_ar, end_ar, program_buf, start_x, proposal->program); // And insert this where we wanted:
				
				float proposed_len = (float) find_program_length(proposal->program);
				float current_len  = (float) find_program_length(current->program);
				
				// forward-back probability
				fb = (-log(current_len)+compute_program_prior(proposal) ) - (-log(proposed_len)+compute_program_prior(current));
			}
			
// 			else if(pp==3) {
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
// 			else if(pp==4) {
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
			compute_posterior(DLEN, device_INPUT, device_OUTPUT, proposal, registers, stack);
			
			// compute whether not we accept the proposal, while rejecting infs and nans
			int swap = (random_float(rx,ry,rz,rw) < exp(proposal->posterior - current->posterior - fb) && is_valid(proposal->posterior)) || !is_valid(current->posterior);
			
			// Use a trick to swap pointers without branching 
			ifthenswap(swap, (void**)&current, (void**)&proposal, swaptmp);
			
		} // end for each proposal kind
	}
	
	if (idx<N) {
// 		int tmp[MAX_PROGRAM_LENGTH] = {1,0,0,1,0,0,1,0,7,10,2,6,1,9,5,0,9,1,2,6,6,1,8,3,10,7,5,10,1,1,0,1,1,3,6,4,2,0,9,5,1,10,1,3,3,0,4,10,6,8,4,2,1,6,10,4,11,4,6,1,5,11,1,11,9,9,3,0,0,1,6,10,4,4,3};
// 		memcpy(current->program, tmp, sizeof(tmp));
// 		compute_posterior(DLEN, device_INPUT, device_OUTPUT, current, registers, stack);
// // 		current->posterior = f_output( 2.0, current, registers, stack); //compute_likelihood(DLEN, device_INPUT, device_OUTPUT, current, registers, stack);  
// 		
// 		float x = 2.0;
// 		current->posterior = sub(mul(mul(sin(pow(1.0f,x)),x),sub(exp(exp(asin(1.0f))),asin(div(1.0f,pow(mul(asin(mul(sin(pow(1.0f,add(mul(log(pow(sin(mul(x,sub(sub(1.0f,sin(1.0f)),div(exp(x),add(mul(pow(sub(1.0f,1.0f),x),1.0f),1.0f))))),sin(div(-sin(sub(log(1.0f),pow(pow(add(1.0f,exp(x)),div(exp(1.0f),pow(add(sin(-x),1.0f),x))),x))),1.0f)))),x),x))),1.0f)),0.f),0.f))))),0.f);
// // 		
		memcpy(out_hypotheses + idx, (void*)current, sizeof(hypothesis));
	}
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
		
		device_run<<<N_BLOCKS,BLOCK_SIZE>>>(N, DLEN, device_INPUT, device_OUTPUT, device_out_hypotheses, seed+N*outer, outer==0);
		
		// Retrieve result from device and store it in host array
		cudaMemcpy(host_out_hypotheses, device_out_hypotheses, out_hypotheses_size, cudaMemcpyDeviceToHost);
		
		// sort the samples by probability:
		qsort( (void*)host_out_hypotheses, N, sizeof(hypothesis), hypothesis_posterior_compare);
		
		//Print results
		for(int n=0; n<N; n++){
			printf("%d\t%d\t%4.2f\t%4.2f\t%4.2f\t%4.2f\t", outer, n, host_out_hypotheses[n].posterior,  host_out_hypotheses[n].prior, host_out_hypotheses[n].likelihood, host_out_hypotheses[n].llsigma);
			
			printf("\"");
			for(int i=0;i<MAX_PROGRAM_LENGTH;i++) 
				printf("%d ", host_out_hypotheses[n].program[i]);
			printf("\"\t");
					
			printf("\"");
			print_program_as_expression( &(host_out_hypotheses[n]) );
			printf("\"\n");
		}
		fflush(stdout);
		
		
// 		TODO: FIX THIS SO THAT IF WE DON'T US ETHIs, It SHOULDN'T FUCK UP THE OHTER RESULTS!!
		
		// Now go through and find the top TOP_RECOPY structurally unique hypotheses
		// and put them at the end
// 		int j = N-1;
// 		for(int i=N;i>=0 && j>=N-TOP_RECOPY;i--) {
// 			int keep = 1;
// 			
// 			for(int chk=j+1;chk<N;chk++){ // check to see if this is identical to anything previous that we've kept
// 				if(hypothesis_structurally_identical(  &(host_out_hypotheses[i]),  &(host_out_hypotheses[chk] )) ) {
// 					keep = 0;
// 					break;
// 				}
// 			}
// 			
// 			if(keep) {
// 				memcpy( (void*)&(host_out_hypotheses[j]), (void*)&(host_out_hypotheses[i]), sizeof(hypothesis)); 
// 				j--; 
// 			}
// 		}
// 		
// //		 We will take the top N and recopy along in the host
// 		for(int i=N-1;i>=N-TOP_RECOPY;i--) { // copy the best, which start from the end
// 			for(int j=i-TOP_RECOPY;j>=0; j -= TOP_RECOPY) // and go backwards
// 				memcpy( (void*)&(host_out_hypotheses[j]), (void*)&(host_out_hypotheses[i]), sizeof(hypothesis)); 
// 			
// 		}
// 		
// 		cudaMemcpy(device_out_hypotheses, host_out_hypotheses, out_hypotheses_size, cudaMemcpyHostToDevice);
		
	}

	// Cleanup
	delete[] host_out_hypotheses;
	delete[] host_OUTPUT;
	delete[] host_INPUT;
	
	cudaFree(device_OUTPUT);
	cudaFree(device_INPUT);
	cudaFree(device_out_hypotheses);
}