/*
 * GPUropolis - 2016 June 29 - Steve Piantadosi 
 * 
 * Main code!
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <getopt.h>
#include <string.h>
#include <vector>
#include "cuPrintf.cu" //!

// not computed on chains but in the actual prior and likelihood:
const float PRIOR_TEMPERATURE = 10.0;
const float LL_TEMPERATURE = 1.0;

// in MCMC only (not on hypotheses)
const float mcmcACCEPTANCE_TEMPERATURE = 1.0;

const int STACK_START = 100; // where do we start the stack (this much up and down allowed)
const int STACK_SIZE  = 200; // how big is the stack?

const int MAX_PROGRAM_LENGTH = 30;
const int MAX_CONSTANTS = 5; // how many constants per hypothesis at most?

#include "structures.cu"
#include "src/misc.cu"
#include "src/data.cu"
#include "src/virtual-machine.cu"
#include "src/hypothesis.cu"
#include "src/MH-simple-kernel.cu"

using namespace std;

int N = 1024;  // Hw many chains?

const int BLOCK_SIZE = 128;
int N_BLOCKS = 0; // set below
const int HARDARE_MAX_X_BLOCKS = 1024;
const int HARDWARE_MAX_THREADS_PER_BLOCK = 1024; // cannot exceed this many threads per block! For compute level 2.x and greater!

string in_file_path = "data.txt"; 
string OUT_PATH     = "run";

int SEED = -1; // Random number seed (for replicability) if -1, we use time()

int MCMC_ITERATIONS = 1000; 
int OUTER_BLOCKS = 1;
int BURN_BLOCKS = 0; // how many blocks (of MCMC_ITERATIONS each) do we burn-in? This is not sensible if we aren't using samples as samples

int FIRST_HALF_DATA = 0; // use only the first half of the data
int EVEN_HALF_DATA  = 0; // use only the even half of the data

int seed;

// the output paths (defined below)
string SAMPLE_PATH, MAP_PATH, LOG_PATH, PERFORMANCE_PATH;

// These are shared arrays among all main functions:
mcmc_results* host_mcmc_results;
mcmc_specification* host_spec;
datum* host_data;
int DATA_LENGTH; // the amount of data

// --------------------------------------------------------------------------------------------------------------
// Run MCMC
// --------------------------------------------------------------------------------------------------------------

void host_run_MCMC(int N, mcmc_specification* host_spec, mcmc_results* host_mcmc_results, datum* host_data) {
	
	// ----------------------
	// Set up all of the bits
	// ----------------------
	
	// and put this on the GPU
	datum* device_data; 
	cudaMalloc((void **) &device_data,  DATA_LENGTH*sizeof(datum));
	cudaMemcpy(device_data, host_data,  DATA_LENGTH*sizeof(datum), cudaMemcpyHostToDevice);
	for(int i=0;i<N;i++) {
		host_spec[i].data_length = DATA_LENGTH;
		host_spec[i].data = device_data;
	}

	// And create and copy these results:
	int MCMC_RESULTS_SIZE = sizeof(mcmc_results)*N;
	mcmc_results* device_mcmc_results; cudaMalloc((void **) &device_mcmc_results, MCMC_RESULTS_SIZE); // device allocate

	int NSPECSIZE = N*sizeof(mcmc_specification);
	mcmc_specification* dev_spec; cudaMalloc((void **) &dev_spec, NSPECSIZE ); // device allocate

	// compute the maximum possible ll
	double PERFECT_LL = 0.0;
	for(int di=0;di<DATA_LENGTH;di++) {PERFECT_LL += lnormalpdf( 0.0, host_data[di].sd); }
	
	
	clock_t mytimer;
	
	// ----------------------
	// Main loop
	// ----------------------
	for(int outer=0;outer<OUTER_BLOCKS+BURN_BLOCKS;outer++) {
		double secDEVICE=0.0, secHOST=0.0, secTRANSFER=0.0; // how long do we spend on each?	
		
		// Set up the rng (also we can anneal here if we want)
		for(int i=0;i<N;i++) {
			host_spec[i].rng_seed = seed + (1+outer)*N*(i+1); // set this seed; i+1 and outer+1 here prevent us getting a zero
		}
		
		// copy to host
		mytimer = clock(); 
		cudaMemcpy(dev_spec, host_spec, NSPECSIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(device_mcmc_results, host_mcmc_results, MCMC_RESULTS_SIZE, cudaMemcpyHostToDevice);
		secTRANSFER = double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		//////////////// Now run: //////////////// 
		////////////////////////////////////////// 
		
		
		cudaPrintfInit(); // set up cuPrintf
        cudaDeviceSynchronize(); 
		
        mytimer = clock();
        MH_simple_kernel<<<N_BLOCKS,BLOCK_SIZE>>>(N, dev_spec, device_mcmc_results);
		cudaDeviceSynchronize(); // wait for preceedings requests to finish
		secDEVICE = double(clock() - mytimer) / CLOCKS_PER_SEC;
        
		cudaPrintfDisplay(stdout, true); // clean up cuPrintf
		cudaPrintfEnd(); //
		
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess) {
			printf("CUDA error: %s\n", cudaGetErrorString(error));
		}		
		
		////////////////////////////////////////// 
		////////////////////////////////////////// 
		
		// Retrieve result from device and store it in host array
		mytimer = clock();
		cudaMemcpy(host_mcmc_results, device_mcmc_results, MCMC_RESULTS_SIZE, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		secTRANSFER += double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		// -----------------------------------------------------------------------------------------------------
		// And translate them to a temporary "hist_hypotheses" array
		
		// For locally manipulating hypotheses -- just for displaying
		mytimer = clock(); // for timing the rest of host operations
		hypothesis* host_hypotheses = new hypothesis[N]; 
		hypothesis* host_out_MAPs   = new hypothesis[N];
	
		for(int i=0;i<N;i++) {
			COPY_HYPOTHESIS( &(host_hypotheses[i]), &(host_mcmc_results[i].sample) );
			COPY_HYPOTHESIS( &(host_out_MAPs[i]),   &(host_mcmc_results[i].MAP) );
		}	
		
		// sort them as required below
		qsort( (void*)host_hypotheses, N, sizeof(hypothesis), hypothesis_posterior_compare);
		qsort( (void*)host_out_MAPs,   N, sizeof(hypothesis), hypothesis_posterior_compare);
		
		// Save results. But not for burn blocks
		if(outer >= BURN_BLOCKS) {
            
            FILE* fp_sample = fopen(SAMPLE_PATH.c_str(), "a");
            FILE* fp_map    = fopen(MAP_PATH.c_str(), "a");
            
            for(int i=0;i<N;i++) {
                dump_to_file(fp_sample, &host_hypotheses[i], i,  outer);
                dump_to_file(fp_map,    &host_out_MAPs[i],   i,  outer);
            }
            
           fclose(fp_sample); 
           fclose(fp_map); 
		}
        
		delete[] host_hypotheses;
		delete[] host_out_MAPs;
		
		// Now how to handle the end of a "block" -- what update do we do?
		secHOST = double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		// -----------------------------------------------------------------------------------------------------
		// output some performance stats (we do this for burn blocks)
		
		double secTOTAL = secHOST + secTRANSFER + secDEVICE;
		
		double AR_mean=0.0; // get the accept/reject mean
		for(int i=0;i<N;i++) 
			AR_mean += float(host_mcmc_results[i].acceptance_count) / float(host_mcmc_results[i].proposal_count);
		AR_mean /= float(N);
		
		FILE* fp = fopen(PERFORMANCE_PATH.c_str(), "a");
		fprintf(fp, "%i\t%.2f\t%.6f\t%.6f\t%.6f\t%.2f\t%.2f\t%.2f\t%.5f\n",
			outer, 
			PERFECT_LL,
			secDEVICE, 
			secTRANSFER, 
			secHOST, 
			double(N)*double(MCMC_ITERATIONS)/ secTOTAL,
			double(N)*double(MCMC_ITERATIONS)*double(DATA_LENGTH)/secTOTAL,
			double(N*MCMC_RESULTS_SIZE)/(1048576. * secTRANSFER),
       		AR_mean
		);
		
		fclose(fp);
	
	} // end outer loop 
	
	// and clean up
	cudaFree(device_data);
	cudaFree(dev_spec);
	cudaFree(device_mcmc_results);
	
}


// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
// main routine that executes on the host
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------------------

static struct option long_options[] =
    {   
        {"in",           required_argument,    NULL, 'd'},
        {"iterations",   required_argument,    NULL, 'i'},
        {"N",            required_argument,    NULL, 'N'},
        {"out",          required_argument,    NULL, 'O'},
        {"outer",        required_argument,    NULL, 'o'},
        {"seed",         required_argument,    NULL, 's'},
        {"burn",         required_argument,    NULL, 'b'},
        {"first-half",   no_argument,    NULL, 'f'},
        {"even-half",    no_argument,    NULL, 'e'},
        {"all",    no_argument,    NULL, '_'},
        {NULL, 0, 0, 0} // zero row for bad arguments
    };  

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
			case 'O': OUT_PATH = optarg; break;
			case 'b': BURN_BLOCKS = atoi(optarg); break;
			case 's': SEED = (float)atof(optarg); break;
			case 'f': FIRST_HALF_DATA = 1; break;
			case 'e': EVEN_HALF_DATA = 1; break;
			case '_': break; // don't do anything if we use all the data
			default: return 1; // unspecified
		}
	
	N_BLOCKS = N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1);
	
	assert(N_BLOCKS < HARDARE_MAX_X_BLOCKS); // can have at most this many blocks
	assert(N/N_BLOCKS <= HARDWARE_MAX_THREADS_PER_BLOCK); // MUST HAVE LESS THREADS PER BLOCK!!
	
	// -----------------------------------------------------------------------
	// Set up the output files etc
	// -----------------------------------------------------------------------
	
	SAMPLE_PATH = OUT_PATH+"/samples.txt";
	MAP_PATH = OUT_PATH+"/MAPs.txt";
	LOG_PATH = OUT_PATH+"/log.txt";
	PERFORMANCE_PATH = OUT_PATH+"/performance.txt";
// 	string H0_PATH          = OUT_PATH+"/h0.txt"; // what were our initial hypotheses?
	
	// -------------------------------------------------------------------------
	// Make the RNG replicable
	
	if(SEED==-1) {
		srand(time(NULL));
		seed = rand();
	} 
	else {  seed = SEED; }

	// -------------------------------------------------------------------------
	// Write the log and performance log
	
	FILE* fp = fopen(LOG_PATH.c_str(), "w");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << LOG_PATH <<"\n"; exit(1);}
	
	fprintf(fp, "-----------------------------------------------------------------\n");
	fprintf(fp, "-- Parameters:\n");
	fprintf(fp, "-----------------------------------------------------------------\n");
	fprintf(fp, "\tInput data path: %s\n", in_file_path.c_str());
	fprintf(fp, "\tOutput path: %s\n", OUT_PATH.c_str());
	fprintf(fp, "\tMCMC Iterations (per block): %i\n", MCMC_ITERATIONS);
	fprintf(fp, "\tBlocks: %i\n", OUTER_BLOCKS);
	fprintf(fp, "\tBurn Blocks: %i\n", BURN_BLOCKS);
	fprintf(fp, "\tN chains: %i\n", N);
	fprintf(fp, "\tseed: %i\n", seed);
	fprintf(fp, "\tMax program length: %i\n", MAX_PROGRAM_LENGTH);
	
	fprintf(fp, "\n\n");
	fclose(fp);
	
	fp = fopen(PERFORMANCE_PATH.c_str(), "w");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << PERFORMANCE_PATH <<"\n"; exit(1);}
	fprintf(fp,   "block\tperfect.ll\tgpu.time\ttransfer.time\tcpu.time\tsamples.per.second\tf.per.second\ttransfer.mb.per.second\tacceptance.ratio\n");
	fclose(fp);
	
	// -----------------------------------------------------------------------
	// Read the data and set up some arrays
	// -----------------------------------------------------------------------
	
	vector<datum>* data_vec = load_data_file(in_file_path.c_str(), FIRST_HALF_DATA, EVEN_HALF_DATA);
	host_data = &((*data_vec)[0]); // can do this with vectors now
	DATA_LENGTH = data_vec->size();
	
	
	// Echo the data we actually run with (post-filtering for even/firsthalf)
	fp = fopen(LOG_PATH.c_str(), "a");
	fprintf(fp, "\n-----------------------------------------------------------------\n");
	fprintf(fp, "-- Data:\n");
	fprintf(fp, "-----------------------------------------------------------------\n");
	for(int i=0;i<DATA_LENGTH;i++) 
		fprintf(fp, "\t%f\t%f\t%f\n", host_data[i].input, host_data[i].output, host_data[i].sd);
	fclose(fp);
	
	
	// -----------------------------------------------------------------------
	// Set up the results and specifications locally
	// -----------------------------------------------------------------------
		
	host_mcmc_results = new mcmc_results[N]; // this holds the hypothese we start and stop with
	host_spec = new mcmc_specification[N]; 

	// We send host_mcmc_results[i].sample as the starting point for each chain. 
    // here, we will initialize randomly. 
	for(int i=0;i<N;i++){
        initialize(&(host_mcmc_results[i].sample), seed);
    }
	
	// -----------------------------------------------------------------------
	// Run
	// -----------------------------------------------------------------------
	
	// Set the specifications
	for(int i=0;i<N;i++) {
		host_spec[i].acceptance_temperature = mcmcACCEPTANCE_TEMPERATURE;
		host_spec[i].iterations = MCMC_ITERATIONS; // how many steps to run?
	}
		
	// run MCMC
	host_run_MCMC(N, host_spec, host_mcmc_results, host_data); // then we still have to run the partially-filled one!
    
	// -----------------------------------------------------------------------
	// Cleanup
	// -----------------------------------------------------------------------
	
	delete[] host_data;
	delete[] host_mcmc_results;
	delete[] host_spec;
}
