/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Main code!
 * 
 * Repetitions -- tops are maintained through reps, but samples and MAPs are distinguishable
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

#include "src/data.cu"
#include "src/misc.cu"
#include "src/primitives.cu"
#include "src/__PRIMITIVES.cu"
#include "src/hypothesis.cu"
#include "src/mcmc-package.cu"
#include "src/virtual-machine.cu"

#include "src/constant-kernel.cu"

using namespace std;

int N = 1024;  // Hw many chains?

const int BLOCK_SIZE = 128;
int N_BLOCKS = 0; // set below
const int HARDARE_MAX_X_BLOCKS = 1024;
const int HARDWARE_MAX_THREADS_PER_BLOCK = 1024; // cannot exceed this many threads per block! For compute level 2.x and greater!

string in_file_path = "data.txt"; 
string OUT_PATH     = "run";

int SEED = -1; // Random number seed (for replicability) if -1, we use time()

int ENUMERATION_DEPTH = 8; //8

int MCMC_ITERATIONS = 1000; 
int OUTER_BLOCKS = 1;
int BURN_BLOCKS = 0; // how many blocks (of MCMC_ITERATIONS each) do we burn-in? This is not sensible if we aren't using samples as samples

int FIRST_HALF_DATA = 0; // use only the first half of the data
int EVEN_HALF_DATA  = 0; // use only the even half of the data

int HYPOTHESIS_I_COUNTER = 0; // what number hypothesis are we on (overall)?
int SAMPLE_BATCH_COUNTER = 0; // how many batches (of N samples) have we run?

int seed;

// the output paths (defined below)
string SAMPLE_PATH, MAP_PATH, TOP_PATH, ALL_TOP_PATH, LOG_PATH, PERFORMANCE_PATH, RESULTS_MATRIX_PATH;

// These are shared arrays among all main functions:
mcmc_package* host_mcmc_package;
mcmc_package* device_mcmc_package;
datum* host_data;
datum* device_data; 
int DATA_LENGTH; // the amount of data
float PERFECT_LL;
int MCMC_PACKAGE_SIZE;

#include "src/results-matrix.cu" // uses a bunch of the above defines
#include "src/io.cu"

static struct option long_options[] =
	{	
		{"in",           required_argument,    NULL, 'd'},
		{"enumeration",           required_argument,    NULL, 'E'},
		{"iterations",   required_argument,    NULL, 'i'},
		{"N",            required_argument,    NULL, 'N'},
		{"out",          required_argument,    NULL, 'O'},
		{"outer",        required_argument,    NULL, 'o'},
		{"temperature",  required_argument,    NULL, 'T'},
		{"seed",         required_argument,    NULL, 's'},
		{"max-program-length",   required_argument,    NULL, 'L'},
		{"end-of-block-action",     required_argument,        NULL, 'm'},
		{"print-top",    required_argument,    NULL, 't'},
		{"burn",         required_argument,    NULL, 'b'},
		{"first-half",   no_argument,    NULL, 'f'},
		{"even-half",    no_argument,    NULL, 'e'},
		{"all",    no_argument,    NULL, '_'},
		{NULL, 0, 0, 0} // zero row for bad arguments
	};  

// --------------------------------------------------------------------------------------------------------------
// Run MCMC
// --------------------------------------------------------------------------------------------------------------

void host_run_MCMC(int N, mcmc_package* packages, datum* host_data) {
	clock_t mytimer;
	
	// ----------------------
	// Main loop
	// ----------------------
	for(int outer=0;outer<OUTER_BLOCKS+BURN_BLOCKS;outer++) {
		double secDEVICE=0.0, secHOST=0.0, secTRANSFER=0.0; // how long do we spend on each?	
		
		// Set up the rng (also we can anneal here if we want)
		for(int i=0;i<N;i++) {
			host_mcmc_package[i].rng_seed = seed + (1+outer)*N + i; // set this seed
			host_mcmc_package[i].chain_index = HYPOTHESIS_I_COUNTER;
			HYPOTHESIS_I_COUNTER++;
		}
		
		// copy these over, containing our initial hypotheses
		mytimer = clock();
		cudaMemcpy(device_mcmc_package, host_mcmc_package, MCMC_PACKAGE_SIZE, cudaMemcpyHostToDevice);
		secTRANSFER += double(clock() - mytimer) / CLOCKS_PER_SEC;
	
		//////////////// Now run: //////////////// 
		////////////////////////////////////////// 
		
		mytimer = clock();
		
		cudaPrintfInit(); // set up cuPrintf
		
		MH_constant_kernel<<<N_BLOCKS,BLOCK_SIZE>>>(N, device_mcmc_package, MCMC_ITERATIONS);
		cudaSynchronizeAndErrorCheck();
		
		cudaPrintfDisplay(stdout, true); // clean up cuPrintf
		cudaPrintfEnd(); //
		cudaSynchronizeAndErrorCheck();
		
		secDEVICE = double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		////////////////////////////////////////// 
		////////////////////////////////////////// 
		
		// Retrieve result from device and store it in host array
		mytimer = clock();
		
		cudaMemcpy(host_mcmc_package, device_mcmc_package, MCMC_PACKAGE_SIZE, cudaMemcpyDeviceToHost);
		cudaSynchronizeAndErrorCheck();
		
		secTRANSFER += double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		// -----------------------------------------------------------------------------------------------------
		// And translate them to a temporary "hist_hypotheses" array
		
		// For locally manipulating hypotheses -- just for displaying
		mytimer = clock(); // for timing the rest of host operations
		
		// Save results. But not for burn blocks
		if(outer >= BURN_BLOCKS) {
			
			save_mcmc_package_to_file(SAMPLE_PATH.c_str(), 1, outer);
		
			// -----------------------------------------------------------------------------------------------------
			// And update the result matrix, with this sample
			// NOTE: THIS USSES THE CURRENTLY FOUND MAP (WHICH IS OKAY IF WE BURN ENOUGH)
			
			GPU_add_rM();
		}
		
		// Now how to handle the end of a "block" -- what update do we do?
		secHOST = double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		// -----------------------------------------------------------------------------------------------------
		// output some performance stats (we do this for burn blocks)
		
		double secTOTAL = secHOST + secTRANSFER + secDEVICE;
				
		double AR_mean=0.0; // get the accept/reject mean
		for(int i=0;i<N;i++) 
			AR_mean += float(host_mcmc_package[i].acceptance_count) / float(host_mcmc_package[i].proposal_count);
		AR_mean /= float(N);
		
		FILE* fp = fopen(PERFORMANCE_PATH.c_str(), "a");
		fprintf(fp, "%i\t%i\t%.2f\t%.2f\t%.6f\t%.6f\t%.2f\t%.2f\t%.2f\t%.5f\n",
			SAMPLE_BATCH_COUNTER, 
			outer, 
			PERFECT_LL,
			secDEVICE, 
			secTRANSFER, 
			secHOST, 
			double(N)*double(MCMC_ITERATIONS)/ secTOTAL,
			double(N)*double(MCMC_ITERATIONS)*double(DATA_LENGTH)/secTOTAL, 
			double(N*MCMC_PACKAGE_SIZE)/(1048576. * secTRANSFER),
       			AR_mean
		);
		
		fclose(fp);
		
	
	
	} // end outer loop 
	
	// ---------------------------------------------------
	// And process the MAPs (after everything!)
	
	save_mcmc_package_to_file(MAP_PATH.c_str(), 0, SAMPLE_BATCH_COUNTER);
	
	// ---------------------------------------------------
	// And write the result matrix out
		
	write_results_matrix();
	
	// ---------------------------------------------------
	// and keep track of how many batches we've run
	
	SAMPLE_BATCH_COUNTER++;
	
	
}


// --------------------------------------------------------------------------------------------------------------
// Enumeration procedure for hypotheses
// --------------------------------------------------------------------------------------------------------------


// Determine any constraints on what can and cannot be allowed in a potential hypothesis
bool check_in_enumeration(hypothesis* h){
	// 	// Do some checks -- should we keep this?
	int found_X = 0; // require X (non-constant)
	int found_CONSTANT = 0; // and C (uses some constants)
	
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++){
		
		op_t oi = h->program[i];
		op_t oip1 = NOOP_;
		if(i < MAX_PROGRAM_LENGTH-1) oip1 = h->program[i+1];
		
		// TODO: CLEAN UP THESE CONSTRAINTS!
		if( oi==NEG_ && (oip1==NEG_ || oip1 == CONSTANT_) ) return 0; // no double negatives		
		if( (oi==ONE_) && (oip1==MUL_ || oip1==DIV_ || oip1==LOG_ || oip1==POW_) ) return 0; // none of these operations make sense
		if( (oi==ZERO_) && (oip1==MUL_ || oip1==DIV_ || oip1==ADD_ || oip1==SUB_ || oip1==LOG_ || oip1==EXP_ || oip1==POW_) ) return 0; // none of these operations make sense
		
		if(h->program[i] == X_) found_X++;
		if(h->program[i] == CONSTANT_) found_CONSTANT++;
		
	}
	
	return ( found_X>0 && found_CONSTANT>0);
}


int enum_hyp_i=0;
int add_and_run( hypothesis* h ) {
	
	if(! check_in_enumeration(h) ) return 1; //  don't use, but keep going

	
	h->structure_prior = compute_structure_prior(h);
		
	// else add:		
	memcpy( (void*)&(host_mcmc_package[enum_hyp_i].sample), (void*)h, sizeof(hypothesis) );
	memcpy( (void*)&(host_mcmc_package[enum_hyp_i].MAP),    (void*)h, sizeof(hypothesis) );
	memcpy( (void*)&(host_mcmc_package[enum_hyp_i].proposal), (void*)h, sizeof(hypothesis) );
	
	enum_hyp_i++;

	if( enum_hyp_i >= N) {
		assert(enum_hyp_i == N);
		
		// And run one
		host_run_MCMC(N, host_mcmc_package, host_data);
		
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(error)); }
		
		// and reset our counter
		enum_hyp_i = 0;
	}

	
	return 1;
}



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
			case 'E': ENUMERATION_DEPTH = atoi(optarg); break;
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
	TOP_PATH = OUT_PATH+"/tops.txt"; // the top of the most recent repetition 
	ALL_TOP_PATH = OUT_PATH+"/all-tops.txt"; // the tops of each repetition are concatenated to this
	LOG_PATH = OUT_PATH+"/log.txt";
	PERFORMANCE_PATH = OUT_PATH+"/performance.txt";
	RESULTS_MATRIX_PATH = OUT_PATH+"/results-matrix.txt";
	
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
	fprintf(fp, "\tSEED: %i\n", seed);
	fprintf(fp, "\tENUMERATION_DEPTH: %i\n", ENUMERATION_DEPTH);
	fprintf(fp, "\tMax program length: %i\n", MAX_PROGRAM_LENGTH);
	
	fprintf(fp, "\n\n");
	fclose(fp);
	
	fp = fopen(PERFORMANCE_PATH.c_str(), "w");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << PERFORMANCE_PATH <<"\n"; exit(1);}
	fprintf(fp, "repetition\tblock\tperfect.ll\tdevice.time\ttransfer.time\thost.time\tsamples.per.second\tf.per.second\tprimitives.per.second\ttransfer.mb.per.second\tacceptance.ratio\ttop.overlap.pct\tduplicate.pct\n");
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
	// Set up the packages and data locally, and copy
	// -----------------------------------------------------------------------
		
	host_mcmc_package = new mcmc_package[N]; // this holds the hypothese we start and stop with
	
	// createa nd copy the data
	cudaMalloc((void **) &device_data,  DATA_LENGTH*sizeof(datum));
	cudaMemcpy(device_data, host_data,  DATA_LENGTH*sizeof(datum), cudaMemcpyHostToDevice);
	
	// Allocate the device
	MCMC_PACKAGE_SIZE = sizeof(mcmc_package)*N;
	cudaMalloc((void **) &device_mcmc_package, MCMC_PACKAGE_SIZE); // device allocate
	
	// compute the maximum possible ll
	PERFECT_LL = 0.0;
	for(int di=0;di<DATA_LENGTH;di++) {PERFECT_LL += lnormalpdf( 0.0, host_data[di].sd); }

	// -----------------------------------------------------------------------
	// And initialie these
	// -----------------------------------------------------------------------
	
	for(int i=0;i<N;i++){
		initialize( &(host_mcmc_package[i].sample) ); // initialize these so that if we don't fill up, 
		initialize( &(host_mcmc_package[i].proposal) );
		initialize( &(host_mcmc_package[i].MAP) );
		host_mcmc_package[i].MAP.posterior = -1./0.; // must initialize this so we over-write
		
		host_mcmc_package[i].data_length = DATA_LENGTH;
		host_mcmc_package[i].data = device_data;
	}
	
	
	// -----------------------------------------------------------------------
	// Initialize the result matrix
	// -----------------------------------------------------------------------
	
	initialize_rM();
	
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	// Enumerate the hypotheses and run
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	// -----------------------------------------------------------------------
	
		
	// To run successively with enumeration (for larger enumeration)
	// This uses host_mcmc_package to store the mcmc results, etc. 
	enumerate_all_programs(ENUMERATION_DEPTH, add_and_run);
	
	// then we still have to run the partially-filled one!
	host_run_MCMC(N, host_mcmc_package, host_data); 
	
	
	// -----------------------------------------------------------------------
	// Cleanup
	// -----------------------------------------------------------------------
	
	delete[] host_data;
	delete[] host_mcmc_package;

	cudaFree(device_data);
 	cudaFree(device_mcmc_package);	
}