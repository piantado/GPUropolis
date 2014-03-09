/*
 * 
 * -- To run on CPU: http://code.google.com/p/gpuocelot/
 * 
 * TODO:
 * 	- Implement MAIN_LOOP
 * 	- Make a "dump to file" function, so we can simulatneously dump our samples, MAPs, and make top use MAPs!
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
// #include <cuda_runtime.h>
// #include <helper_functions.h>
#include <getopt.h>
#include <string.h>
#include <vector>

// Added for non-CUDA version
#include<iostream>
#include<math.h>
#define cudaFree   free

const float PRIOR_TEMPERATURE = 1.0; // the prior temperature
const float LL_TEMPERATURE = 1.0; // the temperature on the likelihood
float POSTERIOR_TEMPERATURE = 1.0;
const double RESAMPLE_TEMPERATURE = 100.0; // when we resample, what temperature do we take the posterior at?

#include "misc.cpp"
#include "hypothesis.cpp"
#include "programs.cpp"
#include "bayes.cpp"
#include "virtual-machine.cpp"
#include "kernel.cpp"
#include "hypothesis-array.cpp"

using namespace std;

int N = 1024*2;  // Hw many chains?

const int BLOCK_SIZE = 256; // WOW 16 appears to be fastest here...
int N_BLOCKS = 0; // set below

string in_file_path = "data.txt"; 

int SEED = -1; // Random number seed (for replicability) if -1, we use time()

int MCMC_ITERATIONS = 1000; 
int OUTER_BLOCKS = 1;
int BURN_BLOCKS = 1; // how many blocks (of MCMC_ITERATIONS each) do we burn-in?

int FIRST_HALF_DATA = 0; // use only the first half of the data
int EVEN_HALF_DATA  = 0; // use only the even half of the data

int PROPOSAL = 0x2; // a binary sum code for each type of proposal: 1: from prior, 2: standard tree-generation, 4: insert/delete moves (both)
int PRINT_TOP = -1; // if -1 we print all, otherwise we print the top this many from each outer-loop


int MAIN_LOOP = 3; // an integer code for 
/* 
 * 1: start anew each outer loop (restart from prior)
 * 2: maintain the same chain (just print the most recent sample)
 * 3: resample via current probability given by RESAMPLE_TEMPERATURE
 * 4: resample from the global top (also using RESAMPLE_TEMPERATURE)
 */
/*
 * TODO: NOT IMPLEMENTED:
 * 5: resample from the top, penalizing by the number of samples already drawn from that hypothesis. So new things of high rank are 
// double MAIN_RESAMPLE_DISCOUNT = 1.0; // the posterior is penalized by this * [the number of chains started here], so that we will explore newer regions of the space preferentially (even if they are not high probability mass). If this is set to 0.0, then we just resample from the real posterior. If it's +inf, we only restart a chain once

// double RESAMPLE_IF_LOWER = 1000.0; // if we are this much lower than the max, we will be resampled from the top. 
*/

int NTOP = 1000; // store this many of the "top" hypotheses (for resampling). TODO: RIGHT NOW, THIS MUST BE LOWER THAN N DUE TO LACK OF CHECKS BELOW


static struct option long_options[] =
	{	
		{"in",           required_argument,    NULL, 'd'},
		{"iterations",   required_argument,    NULL, 'i'},
		{"N",            required_argument,    NULL, 'N'},
		{"outer",        required_argument,    NULL, 'o'},
		{"temperature",  required_argument,    NULL, 'T'},
		{"seed",         required_argument,    NULL, 's'},
		{"proposal",     required_argument,    NULL, 'p'},
		{"main",     required_argument,        NULL, 'm'},
		{"print-top",    required_argument,    NULL, 't'},
		{"burn",         required_argument,    NULL, 'b'},
		{"first-half",   no_argument,    NULL, 'f'},
		{"even-half",    no_argument,    NULL, 'e'},
		{NULL, 0, 0, 0} // zero row for bad arguments
	};  

// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
// main routine that executes on the host
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{	
	
	// ----------------------------------------------------------------------------
	// Parse command line
	// -----------------------------------------------------------------------
	int option_index = 0, opt=0;
	while( (opt = getopt_long( argc, argv, "bp", long_options, &option_index )) != -1 )
		switch( opt ) {
			case 'd': in_file_path = optarg; break;
			case 'i': MCMC_ITERATIONS = atoi(optarg); break;
			case 'N': N = atoi(optarg); break;
			case 'o': OUTER_BLOCKS = atoi(optarg); break;
			case 'b': BURN_BLOCKS = atoi(optarg); break;
			case 'T': POSTERIOR_TEMPERATURE = (float)atof(optarg); break;
			case 's': SEED = (float)atof(optarg); break;
			case 'f': FIRST_HALF_DATA = 1; break;
			case 'm': MAIN_LOOP = atoi(optarg); break;
			case 't': PRINT_TOP = atoi(optarg); break;
			case 'e': EVEN_HALF_DATA = 1; break;
			case 'p': PROPOSAL = atoi(optarg); break;
			case 'h': // help output:
// 				cout << "Options: " << endl;
// 				cout << "\t--max-base=N         sets the maximum base to N" << endl;
// 				cout << "\t--max-power=N        sets the maximum power to N" << endl;
				return 0;
			default:
				return 1;
		}
	
	N_BLOCKS = N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1);
	
	// -----------------------------------------------------------------------
	// Read the data:
	// -----------------------------------------------------------------------
	
	FILE* fp = fopen(in_file_path.c_str(), "r");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << in_file_path <<"\n"; return 1;}
	
	vector<float> inx; float x;
	vector<float> iny; float y;
	vector<float> insd;  float sd;
	char* line = NULL; size_t len=0;
	while( getline(&line, &len, fp) != -1) {
		
		if( line[0] == '#' ) {
			continue;  // skip comments
		}	
		else if (sscanf(line, "%f\t%f\t%f\n", &x, &y, &sd) == 3) { 
			inx.push_back(x);
			iny.push_back(y);
			insd.push_back(sd);
		}
		else {
			cerr << "*** ERROR IN INPUT\t" << line << endl;
			exit(1);
		}
	}
	assert(inx.size() == iny.size());
	assert(inx.size() == insd.size());
	
	// -----------------------------------------------------------------------
	// Trim the data based on first/second half or even odd
	// (Inefficient here but data sizes are small)
	// -----------------------------------------------------------------------
	
	if(FIRST_HALF_DATA){
		int mid = inx.size()/2;
		for(int i=inx.size()-1;i>=mid;i--) {
			inx.erase(inx.begin()+i);
			iny.erase(iny.begin()+i);
			insd.erase(insd.begin()+i);
		}
	}
	if(EVEN_HALF_DATA) {
		for(int i=inx.size()-1;i>=0;i-=2) {
			inx.erase(inx.begin()+i);
			iny.erase(iny.begin()+i);
			insd.erase(insd.begin()+i);		
		}
	}
	
	// -----------------------------------------------------------------------
	// Echo the run data:
	// -----------------------------------------------------------------------
	
// 	for(int i=0;i<inx.size();i++) {
// 		printf("# %4.4f\t%4.4f\t%4.4f\n", inx[i], iny[i], insd[i]);
// 	}
	
	// -----------------------------------------------------------------------
	// Check that our prior normalizes correctly
	// TODO: can't access that array here.
	// -----------------------------------------------------------------------
	
// 	float psum = 0.0;
// 	for(int i=0;i<NUM_OPS;i++) psum += prior[i];
// 	cerr << psum << endl;
// 	assert( abs(1.0-psum) < 1e-4);

	
	// -----------------------------------------------------------------------
	// Set up some bits...
	// -----------------------------------------------------------------------
	
	const int DLEN = inx.size();
	const size_t DATA_BYTE_LEN = DLEN*sizeof(float);
	
	// Make the RNG replicable
	int seed;
	if(SEED==-1) {
		srand(time(NULL));
		seed = rand();
	} 
	else {
		seed = SEED;
	}
	printf("# Started with seed %i\n", seed);

	
	size_t out_hypotheses_size = N * sizeof(hypothesis);
	hypothesis* host_out_hypotheses = new hypothesis[N]; 
	hypothesis* device_out_hypotheses = (hypothesis*)malloc(out_hypotheses_size); // device allocate
	
	hypothesis* host_out_MAPs = new hypothesis[N];
	hypothesis* host_hypothesis_tmp = new hypothesis[N]; 
	hypothesis* device_out_MAPs = (hypothesis*)malloc(out_hypotheses_size); // device allocate
	
	// store the top hypotheses overall
	hypothesis* host_top_hypotheses = new hypothesis[NTOP+N]; // store N so that we can push N on the end, sort, and then remove duplicates
	hypothesis blankhyp; blankhyp.posterior = -1.0/0.0;// a hypothesis that is blank, for filling up host_top_hypotheses
	for(int i=0;i<NTOP+N;i++) memcpy( (void*)&host_top_hypotheses[i], &blankhyp, sizeof(hypothesis)); // fill up host_top with blanks
	
	// -----------------------------------------------------------------------
	// copy the read input to the GPU
	// -----------------------------------------------------------------------
	
	float* device_INPUT =(float*) malloc(DATA_BYTE_LEN);
	float* host_INPUT = (float*) malloc(DATA_BYTE_LEN);
	for(int i=0;i<DLEN;i++) host_INPUT[i] = inx[i];
	memcpy(device_INPUT, host_INPUT, DATA_BYTE_LEN);
	
	float* device_OUTPUT = (float*)malloc(DATA_BYTE_LEN);
	float* host_OUTPUT = (float*) malloc(DATA_BYTE_LEN);
	for(int i=0;i<DLEN;i++) host_OUTPUT[i] = iny[i];
	memcpy(device_OUTPUT, host_OUTPUT, DATA_BYTE_LEN);
	
	float* device_SD = (float*)malloc(DATA_BYTE_LEN);
	float* host_SD = (float*) malloc(DATA_BYTE_LEN);
	for(int i=0;i<DLEN;i++) host_SD[i] = insd[i];
	memcpy(device_SD, host_SD, DATA_BYTE_LEN);
	
	// -----------------------------------------------------------------------
	// Do calculation on device:
	// -----------------------------------------------------------------------
	
	for(int outer=0;outer<OUTER_BLOCKS+BURN_BLOCKS;outer++) {
		
		// -----------------------------------------------------------------------------------------------------
		// Run
		device_run(N, PROPOSAL, MCMC_ITERATIONS, POSTERIOR_TEMPERATURE, DLEN, device_INPUT, device_OUTPUT, device_SD, device_out_hypotheses, device_out_MAPs, seed+N*outer,  (outer==0)||(MAIN_LOOP==1) );
		
// 		cudaDeviceSynchronize(); // wait for all preceedings requests to finish
			
		// Retrieve result from device and store it in host array
		memcpy(host_out_hypotheses, device_out_hypotheses, out_hypotheses_size);
		memcpy(host_out_MAPs, device_out_MAPs, out_hypotheses_size);
		
		// -----------------------------------------------------------------------------------------------------
		// make sure our check bits have not changed -- that we didn't overrun anything
		
		for(int rank=0; rank<N; rank++){
			assert(host_out_hypotheses[rank].check0 == 33);
			assert(host_out_hypotheses[rank].check1 == 33);
			assert(host_out_hypotheses[rank].check2 == 33);
			assert(host_out_hypotheses[rank].check3 == 33);			
		}

		// -----------------------------------------------------------------------------------------------------
		// Print results -- not for burn blocks
		
		if(outer >= BURN_BLOCKS) {
			
			int start = (PRINT_TOP==-1) ? 0 : N-PRINT_TOP;
			
			// If we want to print MAPs:
			// sort the samples by probability:
// 			qsort( (void*)host_out_MAPs, N, sizeof(hypothesis), hypothesis_posterior_compare);
// 			for(int rank=start; rank<N; rank++){
// 				print_hypothesis(outer, rank, 1, 0, &(host_out_MAPs[rank]));
// 			}
			
			//Print results
			qsort( (void*)host_out_hypotheses, N, sizeof(hypothesis), hypothesis_posterior_compare);
			for(int rank=start; rank<N; rank++){
				print_hypothesis(outer, rank, 0, 0, &(host_out_hypotheses[rank]));
			}
			fflush(stdout);		
		}
		
		// -----------------------------------------------------------------------------------------------------
		// Manage our collection of the top hypotheses (here: samples, not MAPs)
		// we put these in host_top_hypotheses, and the first NTOP of these always stores the best
		// we maintain this by copying all of host_out_hypotheses to the end, resorting, and removing duplicates
		
		memcpy( (void*)&host_top_hypotheses[NTOP+1], (void*)host_out_hypotheses, out_hypotheses_size); // put these at the end
		qsort(  (void*)host_top_hypotheses, NTOP+N, sizeof(hypothesis), neghypothesis_posterior_compare); // resort, best first
		for(int i=0,j=0;i<NTOP;i++,j++) {
			
			if(j<NTOP+N) {
				while( hypothesis_structurally_identical(&host_top_hypotheses[j], &host_top_hypotheses[j+1])){
					j++;				
				}
				if(j!=i){
					memcpy( (void*)&host_top_hypotheses[i], &host_top_hypotheses[j], sizeof(hypothesis));
				}
			}
			else { // out of hyppotheses, so pad with blankhyp
				memcpy( (void*)&host_top_hypotheses[i], &blankhyp, sizeof(hypothesis));
				
			}
		}
		
// 		for(int rank=0; rank<NTOP; rank++){
// 			print_hypothesis(-1, rank, 0, 0, &(host_top_hypotheses[rank]));
// 		}
// 		
		
		// -----------------------------------------------------------------------------------------------------
		// Now how to handle the end of a "block" -- what update do we do?
		if(MAIN_LOOP == 1) {
			// Then we just regenerate. This happens above by passing the last variable to device_run
		}
		else if(MAIN_LOOP == 2) {
			// Then we do nothing. Continue the same chain
		}
		else if(MAIN_LOOP == 3) {
			
			// Resample from the current chain
			multinomial_sample(N, host_hypothesis_tmp, host_out_hypotheses, N, RESAMPLE_TEMPERATURE);
		
			// Since we modified, copy back to the device array
// 			cudaMemcpy(device_out_hypotheses, host_hypothesis_tmp, out_hypotheses_size, cudaMemcpyHostToDevice);
		}
		else if(MAIN_LOOP == 4) {
		
			// resample from the top!
			multinomial_sample(N, host_hypothesis_tmp, host_top_hypotheses, NTOP, RESAMPLE_TEMPERATURE);
		
			// Since we modified, copy back to the device array
// 			cudaMemcpy(device_out_hypotheses, host_hypothesis_tmp, out_hypotheses_size, cudaMemcpyHostToDevice);
		}
	}
	
	// -----------------------------------------------------------------------
	// Cleanup
	// -----------------------------------------------------------------------
	
	delete[] host_out_hypotheses;
	delete[] host_hypothesis_tmp;
	delete[] host_OUTPUT;
	delete[] host_INPUT;
	delete[] host_SD;
	
	cudaFree(device_OUTPUT);
	cudaFree(device_INPUT);
	cudaFree(device_SD);
	cudaFree(device_out_hypotheses);
}