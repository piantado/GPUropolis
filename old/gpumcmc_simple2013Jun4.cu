/*
 * TODO:
 * 	- Make the Stack twice as long as program, start in the middle, and fill up with 0s initially (so we can execute all ADDs, for instance)
 * 	- OTHER GOOD MUTATIONS -- INSERT AND DELETE
 * 	- We can examine the "health" of the chain by seeing how many times we find the "top" one
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>

// this is our arithmetic if/then macro
#define ifthen(x,y,z) ((y)*(x) + (z)*(1-(x)))

const int N = 1000;  // How many chains?
const int BLOCK_SIZE = 128;
const int N_BLOCKS = N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1);
	
const int   DLEN         = 7; // how long is the data?
__constant__ const float INPUT[DLEN]  = { 1.0, 2.0, 3.0, 4.0,5.0, 6.0, 7.0 };
// __constant__ const float OUTPUT[DLEN] = { 0.000000000, -0.6065307, -1.4330626, -2.3364023, -3.2749230, -4.2324086, -5.2012674 }; // (1-x) / exp(1/x)
__constant__ const float OUTPUT[DLEN] = { 0.0, 0.4041569,0.9990636,1.5557204,2.0408044,2.4605594,2.8264565}; // log(1+x*(x-1)) / (exp(2/x))

const float LL_SIGMA     = 0.05;
const float PRIOR_TEMPERATURE = 1.00;

const int MCMC_ITERATIONS = 5000; //50000;
const float RESAMPLE_P = 0.2; // when we propose, with what probability do we change the program?

const int MAX_PROGRAM_LENGTH = 15; 

// CONSTANTS, 2-ARY, 1-ARY
// NUM_OPS stores the number of opsn
// TODO: IF YOU CHANGE THESE, MYOU MSUT CHANGE THE CONDITION BELOW FOR DECODING WHAT HAPPENS TO THE TOP OF STACK
enum OPS { ZERO=0, ONE=1, X=2,    ADD=3, SUB=4, MUL=5, DIV=6,     LOG=7, EXP=8, NUM_OPS=9 };
//           0      1      2       3      4      5      6           7     8       9
// __constant const float OP_ARGS[NUM_OPS] = 


/* "0 0 0 5 0 2 6 0 1 8 8 2 6 8 2 6 7 2 7 5 
 *  x/exp(x/exp(exp(1))), 
 *  
 * 
 */

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

__device__ float f_output(float x, int* program, float* registers, float* stack) {

	// first zero the stack
	for(int i=0;i<2*MAX_PROGRAM_LENGTH;i++) stack[i] = 0.0; 
	
	registers[ZERO] = 0.0f;
	registers[ONE]  = 1.0f;
	registers[X]    = x;
	
	int top = MAX_PROGRAM_LENGTH; //  We start in the middle of the stack
	for(int p=0;p<MAX_PROGRAM_LENGTH;p++) { // program pointer
		int op = program[p];
		
		// update the virtual registers
		registers[ADD]  = stack[top] + stack[top-1];
		registers[SUB]  = stack[top] - stack[top-1];
		registers[MUL]  = stack[top] * stack[top-1];
		registers[DIV]  = stack[top] / stack[top-1];
		registers[LOG]  = log(stack[top]);
		registers[EXP]  = exp(stack[top]);
		
		// what elements changes? If we have a constant, we push; a 1-ary and we replace; a 2-ary and we gobble one
		top = (op<=X)*(top+1) + (op>X && op<=DIV)*(top-1) + (op>DIV)*(top); // TODO: CHECK THIS
		stack[top] = registers[op];
	}
	
	return stack[top];
}

__device__ float compute_likelihood(int* program, float* registers, float* stack) {

	float ll = 0.0;
	for(int i=0;i<DLEN;i++){
		// compute the difference between the output and what we see
		float d = OUTPUT[i] - f_output( INPUT[i], program, registers, stack);
		
		// and use a gaussian likelihood
		ll += (-d*d)/(2.0*LL_SIGMA*LL_SIGMA) - 0.5 * log(2.0*3.141592653589*LL_SIGMA*LL_SIGMA);
	}
	return ll;
}

// Compute the program's prior, 2^-length, but note that the program must be pushed to the "right" in the array,
// so we can do this by counting the number of leading zeros (TODO: WHICH ISN'T QUITE RIGHT)
__device__ float compute_prior(int* program) {
	int keep = 1;
	int counter = 0;
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
		keep = (keep && program[i] == ZERO);
		counter += keep;
	}
	return float(-(MAX_PROGRAM_LENGTH-counter)/PRIOR_TEMPERATURE);
}

__device__ float compute_posterior(int* program, float* registers, float* stack) {
	return compute_prior(program) + compute_likelihood(program, registers, stack);
}



// put into target a random expression that closes parens
// __device__ void random_closed_expression(int* target) {
// 	
// }

// Kernel that executes on the CUDA device
__global__ void device_run(int N, int seed, float* out_posteriors, int* out_programs)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	float stack[2*MAX_PROGRAM_LENGTH]; // Stack is twice as big as program, and populated with 0s so we can eval, e.g., a whole prog of ADDs
	float registers[NUM_OPS];
	
	int program_[MAX_PROGRAM_LENGTH];
	int* program = &(program_[0]); // so we have a pointer
	
	int program_proposal_[MAX_PROGRAM_LENGTH];
	int* program_proposal = &(program_proposal_[0]);
	
	// zero out the program to begin
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) program[i] = 0x0;
		
	// Set up for the RNG
	int rx = idx*seed; // 123456789; // MUST be 32 bit
	int ry = 362436069;
	int rz = 521288629;
	int rw = 88675123;
	
	float current_lp = compute_posterior(program, registers, stack);
	
	for(int mcmci=0;mcmci<MCMC_ITERATIONS;mcmci++) {
		
		for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
			int r = random_float(rx,ry,rz,rw) < RESAMPLE_P;
			program_proposal[i] = ifthen(r, random_int(NUM_OPS,rx,ry,rz,rw), program[i]);
		}
		
		float proposed_lp = compute_posterior(program_proposal, registers, stack);
		
		// compute whether not we accept the proposal. Here, the kernel is symmetric
		// so we can ignore it!
		int swap = (random_float(rx,ry,rz,rw) < exp(proposed_lp - current_lp)) & (!isnan(proposed_lp));
		
		// TODO: HMM HERE SWAPPING POIINTERS SEEMS TO BE HARD USING OUR MULT TRICK
		// since you can't multiply pointers (even times zero!)
		// so for now we'll just conditionally copy the arrays
		for(int i=0;i<MAX_PROGRAM_LENGTH;i++) 
			program[i] = ifthen(swap, program_proposal[i], program[i]);
		
		current_lp = ifthen(swap, proposed_lp, current_lp);
	}
	
	if (idx<N) {
		out_posteriors[idx] = current_lp;
		memcpy(out_programs + idx*MAX_PROGRAM_LENGTH, program, MAX_PROGRAM_LENGTH * sizeof(int));
	}
}
 

//------------
// For easier displays
const int MAX_OP_LENGTH = 100; // how much does each string add at most?
char SS[MAX_PROGRAM_LENGTH*2][MAX_OP_LENGTH*MAX_PROGRAM_LENGTH]; 
void print_program_as_expression(int* program) {
	
	char buf[MAX_PROGRAM_LENGTH*MAX_PROGRAM_LENGTH];
	
	int top = MAX_PROGRAM_LENGTH; // top of the stack
	
	// re-initialize our buffer
	for(int r=0;r<MAX_PROGRAM_LENGTH*2;r++) strcpy(SS[r], "0"); // since everything initializes to 0
	
	for(int p=0;p<MAX_PROGRAM_LENGTH;p++) {
		int op = program[p];
		
		switch(op) {
			case ZERO: 
				top += 1;
				strcpy(SS[top], "0");
				break;
			case ONE: 
				top += 1;
				strcpy(SS[top], "1");
				break;
			case X:
				top += 1;
				strcpy(SS[top], "x");
				break;
			case ADD:
				strcpy(buf, "(+ ");
				strcat(buf, SS[top]);
				strcat(buf, " ");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case SUB:
				strcpy(buf, "(- ");
				strcat(buf, SS[top]);
				strcat(buf, " ");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case MUL:
				strcpy(buf, "(* ");
				strcat(buf, SS[top]);
				strcat(buf, " ");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case DIV:
				strcpy(buf, "(/ ");
				strcat(buf, SS[top]);
				strcat(buf, " ");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case LOG:
				strcpy(buf, "(log ");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
				
			case EXP:
				strcpy(buf, "(exp ");
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
	size_t posterior_size = N * sizeof(float);
	float* host_posterior = new float[posterior_size]; //(float*) malloc(posterior_size); 
	
	size_t out_programs_size	 = N * MAX_PROGRAM_LENGTH * sizeof(int);
	int* host_out_programs = new int[out_programs_size]; // (int*) malloc(out_programs_size);
// 	for(int i=0;i<out_programs_size;i++) host_out_programs[i] = 0;
	
	float* device_posterior;
	cudaMalloc((void **) &device_posterior, posterior_size); // device allocate
	
	int* device_out_programs;
	cudaMalloc((void **) &device_out_programs, out_programs_size); // device allocate
	
	// Do calculation on device:
	srand(time(NULL));
	int seed = rand();
	device_run<<<N_BLOCKS,BLOCK_SIZE>>>(N, seed, device_posterior, device_out_programs);
	
	// Retrieve result from device and store it in host array
	cudaMemcpy(host_posterior,    device_posterior,    posterior_size,    cudaMemcpyDeviceToHost);
	cudaMemcpy(host_out_programs, device_out_programs, out_programs_size, cudaMemcpyDeviceToHost);
	
	// Print results
	for(int n=0; n<N; n++){
		printf("%d\t%4.2f\t\"", n, host_posterior[n]);
		print_program_as_expression( host_out_programs + n*MAX_PROGRAM_LENGTH );
// 		for(int i=0;i<MAX_PROGRAM_LENGTH;i++)
// 			printf("%d ", host_out_programs[n*MAX_PROGRAM_LENGTH + i]);
		printf("\"\n");
	}
	
	// Cleanup
	delete[] host_posterior;
	delete[] host_out_programs;
	
	cudaFree(device_posterior);
	cudaFree(device_out_programs);
}