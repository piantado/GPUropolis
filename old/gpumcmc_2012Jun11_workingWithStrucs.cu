/*
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
char* INPUT_FILE = "datasets/squared.txt"; //"datasets/squared.txt";
// __constant__ const float OUTPUT[DLEN] = { 0.0, 0.4041569,0.9990636,1.5557204,2.0408044,2.4605594,2.8264565}; // log(1+x*(x-1)) / (exp(2/x))

const float LL_SIGMA     = 0.10; //0.01;
const float PRIOR_TEMPERATURE = 0.1000;
const int MCMC_ITERATIONS = 1000;

const float RESAMPLE_P = 0.15; // when we propose, with what probability do we change each program element?

const int MAX_PROGRAM_LENGTH = 20; 

// CONSTANTS, 2-ARY, 1-ARY
// NUM_OPS stores the number of opsn
// TODO: IF YOU CHANGE THESE, MYOU MSUT CHANGE THE CONDITION BELOW FOR DECODING WHAT HAPPENS TO THE TOP OF STACK
enum OPS                              { ZERO, ONE, X, PI, E, ADD, SUB, MUL, DIV, POW,     LOG, EXP, NUM_OPS };
__constant__ int NARGS[NUM_OPS]     = {   0,   0,  0,  0, 0,  2,   2,   2,   2,   2,        1,  1 }; // how many args for each op?
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
	int program[MAX_PROGRAM_LENGTH];
} hypothesis;


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

__device__ float f_output(float x, int* program, float* registers, float* stack) {

	// first zero the stack
	for(int i=0;i<2*MAX_PROGRAM_LENGTH;i++) stack[i] = 0.0; 
	
	registers[ZERO] = 0.0f;
	registers[ONE]  = 1.0f;
	registers[X]    = x;
	registers[PI]   = 3.14159;
	registers[E]    = 2.71828;
	
	int top = MAX_PROGRAM_LENGTH; //  We start in the middle of the stack
	for(int p=0;p<MAX_PROGRAM_LENGTH;p++) { // program pointer
		int op = program[p];
		
		// update the virtual registers
		registers[ADD]  = stack[top] + stack[top-1];
		registers[SUB]  = stack[top] - stack[top-1];
		registers[MUL]  = stack[top] * stack[top-1];
		registers[DIV]  = stack[top] / stack[top-1];
		registers[POW]  = pow(stack[top], stack[top-1]);
		registers[LOG]  = log(stack[top]);
		registers[EXP]  = exp(stack[top]);
		
		// the *(op<NUM_OPS&&op>=0) here makes us handle accidentally huge numbers, treating them as 0s
		// what elements changes? If we have a constant, we push; a 1-ary and we replace; a 2-ary and we gobble one
		top += stack_change(op)*(op<NUM_OPS&&op>=0); // so 0 args push, 1 args make no change, 2args eat the top
		stack[top] = registers[op*(op<NUM_OPS&&op>=0)];
	}
// 	
	return stack[top];
}

__device__ float compute_likelihood(int DLEN, float* device_INPUT, float* device_OUTPUT, int* program, float* registers, float* stack) {

	float ll = 0.0;
	for(int i=0;i<DLEN;i++){
		// compute the difference between the output and what we see
		float d = device_OUTPUT[i] - f_output( device_INPUT[i], program, registers, stack);
		
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
	return float(-(MAX_PROGRAM_LENGTH-counter))/float(PRIOR_TEMPERATURE);
}

__device__ void compute_posterior(int DLEN, float* device_INPUT, float* device_OUTPUT, hypothesis* h, float* registers, float* stack) {
	h->prior      = compute_prior(h->program);
	h->likelihood = compute_likelihood(DLEN, device_INPUT, device_OUTPUT, h->program, registers, stack);
	h->posterior = h->prior + h->likelihood;
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
		int newop = random_int(NUM_OPS,x,y,z,w);
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

// Kernel that executes on the CUDA device
__global__ void device_run(int N, int DLEN, float* device_INPUT, float* device_OUTPUT, hypothesis* out_hypotheses, int seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Set up for the RNG
	int rx = idx*seed; // 123456789; // MUST be 32 bit //!!                    <- TODO: HERE WE UNDID THE RANDOM SEED
	int ry = 362436069;
	int rz = 521288629;
	int rw = 88675123;
	
	float stack[2*MAX_PROGRAM_LENGTH]; // Stack is twice as big as program, and populated with 0s so we can eval, e.g., a whole prog of ADDs
	float registers[NUM_OPS];
	
	hypothesis current_; hypothesis* current = &current_;
	hypothesis proposal_; hypothesis* proposal = &proposal_;
	
	int program_buf_[MAX_PROGRAM_LENGTH]; // needed below as a buffer program
	int* program_buf = &(program_buf_[0]);
	
	void* swaptmp[2]; // used for swapping without arithmetic
		
	// randomly initialize
	random_closed_expression(current->program,  rx,ry,rz,rw);

	compute_posterior(DLEN, device_INPUT, device_OUTPUT, current, &(registers[0]), &(stack[0]));

	for(int mcmci=0;mcmci<MCMC_ITERATIONS;mcmci++) {
		
		
		for(int i=0;i<2;i++) { // for each kind of proposal

			float fb = 0.0;  // the forward-backward probability
			int len = find_program_length(current->program); // TODO: CHECK OFF BY !
			
			if(i==0) { // a normal tree regeneration proposal
				
				// first find a proposal location:
				int end_ar = (MAX_PROGRAM_LENGTH-1) - random_int(len-1, rx,ry,rz,rw); // the position of what we replace
				int start_ar = find_close_backwards(current->program, end_ar);
				
				// generate a novel tree in program_buf
				int start_x = random_closed_expression(program_buf, rx,ry,rz,rw);
				special_splice(current->program, start_ar, end_ar, program_buf, start_x, proposal->program); // And insert this where we wanted:
				
				float proposed_len = (float) find_program_length(proposal->program);
				float current_len = (float) find_program_length(current->program);
				
				// forward-back probability
				fb = (-log(current_len)-proposed_len*log(NUM_OPSf)) - (-log(proposed_len)-current_len*log(NUM_OPSf));
			}
			else if(i==1) { // full regeneration proposal
				
				random_closed_expression(proposal->program, rx, ry, rz, rw);
				fb = 0.0; // for using just generate random proposals
			}
			else if(i==2) {
				
				// Simple stupid proposal that just mutate at random:	 
				for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
					int r = random_float(rx,ry,rz,rw) < RESAMPLE_P;
					proposal->program[i] = ifthen(r, random_int(NUM_OPS,rx,ry,rz,rw), current->program[i]);
				}
				//fb = 0.0;
			}
			else if(i==3) {
				
				/// TODO: ON THESE FIX F/B
				
				int pos = (MAX_PROGRAM_LENGTH-1) - random_int(len-1, rx,ry,rz,rw);
				
				// random insert into a program
				for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
					proposal->program[i] = current->program[i+1]*(i<pos) + random_int(NUM_OPS,rx,ry,rz,rw)*(i==pos) + current->program[i]*(i>pos);
				}
				
				fb = (-log(float(len)) - log(NUM_OPSf) ) - (-log(float(len+1)) );
				
			}
			else if(i==4) {
				int pos = (MAX_PROGRAM_LENGTH-1) - random_int(len-1, rx,ry,rz,rw);
				
				// random delete from a program
				proposal->program[0] = 0x0;				
				for(int i=1;i<MAX_PROGRAM_LENGTH;i++) {
					proposal->program[i] = current->program[i-1]*(i<=pos) + current->program[i]*(i>pos);
				}
				
				fb = (-log(float(len)) ) - (-log(float(len-1)) - log(NUM_OPSf) );
				
			}
			
			// compute the posterior for the proposal
			compute_posterior(DLEN, device_INPUT, device_OUTPUT, proposal, &(registers[0]), &(stack[0]));
			
			// compute whether not we accept the proposal, while rejecting infs and nans
			int swap = (random_float(rx,ry,rz,rw) < exp(proposal->posterior - current->posterior - fb) && is_valid(proposal->posterior)) || (! is_valid(current->posterior));
			
			// Use a trick to swap pointers
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
			case PI:
				top += 1;
				strcpy(SS[top], "PI");
				break;
			case E:
				top += 1;
				strcpy(SS[top], "E");
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
			case POW:
				strcpy(buf, "(pow ");
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
	printf("# Starting running\n");
	device_run<<<N_BLOCKS,BLOCK_SIZE>>>(N, DLEN, device_INPUT, device_OUTPUT, device_out_hypotheses, seed);
	printf("# Done running\n");
	
	// Retrieve result from device and store it in host array
	cudaMemcpy(host_out_hypotheses, device_out_hypotheses, out_hypotheses_size, cudaMemcpyDeviceToHost);
	
	//Print results
	for(int n=0; n<N; n++){
		printf("%d\t%4.2f\t%4.2f\t%4.2f\t\"", n, host_out_hypotheses[n].prior+host_out_hypotheses[n].likelihood,  host_out_hypotheses[n].prior, host_out_hypotheses[n].likelihood);
		
		for(int i=0;i<MAX_PROGRAM_LENGTH;i++) printf("%d ", host_out_hypotheses[n].program[i]);
		
		printf("\"\t\"");
		
		print_program_as_expression( host_out_hypotheses[n].program );
		
		printf("\"\n");
	}
	
// 	// Cleanup
// 	delete[] host_out_hypotheses;
// 	delete[] host_OUTPUT;
// 	delete[] host_INPUT;
// 	
// 	cudaFree(device_OUTPUT);
// 	cudaFree(device_INPUT);
// 	cudaFree(device_out_hypotheses);
}