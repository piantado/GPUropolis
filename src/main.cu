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

// not computed on chains but in the actual prior and likelihood:
const float PRIOR_TEMPERATURE = 1.0;
const float LL_TEMPERATURE = 1.0;

// on mcmc specification:
const float ACCEPTANCE_TEMPERATURE = 1.0;

// A new prior to penalize extra deep X ( allowing many compositions on constants, not variables)
const float X_DEPTH_PENALTY = 100.0; // extra penalty for X depth. 0 here gives PCFG generation probability prior
const float X_PENALTY = 10.0; // extra penalty for using X

// Specification of the prior
// in tree resampling, the expected length here is important in getting a good acceptance rate -- too low
// (meaning too long) and we will reject almost everything
const float EXPECTED_LENGTH = 5.0; // also the expected length of proposals
const float PRIOR_XtoCONSTANT = 0.5; //what proportion of constant proposals are x (as opposed to all other constants)?

#include "src/misc.cu"
#include "src/__PRIMITIVES.cu"
#include "src/data.cu"
#include "src/hypothesis.cu"
#include "src/constants.cu"
#include "src/programs.cu"
#include "src/mcmc-specification.cu"
#include "src/mcmc-results.cu"
#include "src/virtual-machine.cu"
#include "src/hypothesis-array.cu"

#include "src/kernels/MH-constant-kernel.cu"
#include "src/kernels/MH-prior-kernel.cu"
#include "src/kernels/MH-simple-kernel.cu"
#include "src/kernels/MH-adaptive-temperature.cu"
#include "src/kernels/MH-adaptive-acceptance-rate.cu"
#include "src/kernels/MH-detect-local-maxima.cu"

using namespace std;

int N = 1024;  // Hw many chains?

const int BLOCK_SIZE = 128;
int N_BLOCKS = 0; // set below
const int HARDARE_MAX_X_BLOCKS = 1024;
const int HARDWARE_MAX_THREADS_PER_BLOCK = 1024; // cannot exceed this many threads per block! For compute level 2.x and greater!

string in_file_path = "data.txt"; 
string OUT_PATH     = "run";

int SEED = -1; // Random number seed (for replicability) if -1, we use time()

int ENUMERATION_DEPTH = 8;

int MCMC_ITERATIONS = 1000; 
int OUTER_BLOCKS = 1;
int BURN_BLOCKS = 0; // how many blocks (of MCMC_ITERATIONS each) do we burn-in? This is not sensible if we aren't using samples as samples

int FIRST_HALF_DATA = 0; // use only the first half of the data
int EVEN_HALF_DATA  = 0; // use only the even half of the data

int seed;

// the output paths (defined below)
string SAMPLE_PATH, MAP_PATH, TOP_PATH, ALL_TOP_PATH, LOG_PATH, PERFORMANCE_PATH;


// These are shared arrays among all main functions:
mcmc_results* host_mcmc_results;
mcmc_specification* host_spec;
datum* host_data;
int DATA_LENGTH; // the amount of data


static struct option long_options[] =
	{	
		{"in",           required_argument,    NULL, 'd'},
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
// Set the prior
// --------------------------------------------------------------------------------------------------------------

void define_prior() {

	assert(NUM_OPS < MAX_NUM_OPS);  // check that we don't have too many for our array

	// how many times have we seen each number of args?
	int count_args[] = {0,0,0};
	for(int i=1;i<NUM_OPS;i++){  // skip NOOP
		assert( hNARGS[i] <= 2); // we must have this to compute expected lengths correctly. It can be changed for arbitrary-arity later
		count_args[ hNARGS[i]]++; 
	}
	
	/*
	* The expected length satisfies:
	* E = p0arg + p1arg(E+1) + p2arg(2 E + 1)
	* E = p0arg+p1arg+p2arg + E ( p1arg + 2 p2arg)
	* E = 1 + E (p1arg + 2 p2arg)
	* 1 = E (1-p1arg-p2arg)
	* so
	* E = 1/(1-p1arg - 2 p2arg)
	* 
	* Constraining p1arg = p2arg,
	* E = 1/(1-3p1arg)
	* 
	* so
	* 
	* p1arg = p2arg = (1-1/E)/3
	* and then we must account for the number in each class
	*/
	float P = (1.0-1.0/EXPECTED_LENGTH)/3.0;
	float P_0arg = (1.0-2.0*P);
	float P_X        = P_0arg * PRIOR_XtoCONSTANT;
	float P_CONSTANT = P_0arg * (1.0-PRIOR_XtoCONSTANT) / float(count_args[0]-1);
	// This way will divide evenly between 1- and 2- args
// 	float P_1arg  = P / float(count_args[1]);
// 	float P_2arg  = P / float(count_args[2]);
	// This way will put all mass equally among all functions, regardless of arity:
	float P_1arg  = 2.*P / float(count_args[1] + count_args[2]);
	float P_2arg  = 2.*P / float(count_args[1] + count_args[2]);;
	
	for(int i=0;i<MAX_NUM_OPS;i++) hPRIOR[i] = 0.0; // must initialize since not all will be used
	
	for(int i=0;i<NUM_OPS;i++) {
		if( i == NOOP_ )         { hPRIOR[i] = 0.0; }
		else if( i == X_ )       { hPRIOR[i] = P_X; }
		else if( hNARGS[i] == 0) { hPRIOR[i] = P_CONSTANT; }
		else if( hNARGS[i] == 1) { hPRIOR[i] = P_1arg; }
		else if( hNARGS[i] == 2) { hPRIOR[i] = P_2arg; }	
	}

	
	// normalize the prior
	double priorZ = 0.0;
	for(int i=0;i<NUM_OPS;i++) priorZ += hPRIOR[i];
	assert(abs(1.-priorZ) < 1e-3); // assert we computed the prior correctly!
	for(int i=0;i<NUM_OPS;i++) hPRIOR[i] /= priorZ;

}

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
	
	
	// and copy PRIOR over to the device
	cudaMemcpyToSymbol(dPRIOR, hPRIOR, MAX_NUM_OPS*sizeof(float), 0, cudaMemcpyHostToDevice);

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
			host_spec[i].rng_seed = seed + (1+outer)*N*i; // set this seed
		}
		
		mytimer = clock();
		cudaMemcpy(dev_spec, host_spec, NSPECSIZE, cudaMemcpyHostToDevice);
		
		// copy these over, containing our initial hypotheses
		cudaMemcpy(device_mcmc_results, host_mcmc_results, MCMC_RESULTS_SIZE, cudaMemcpyHostToDevice);
		secTRANSFER += double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		//////////////// Now run: //////////////// 
		////////////////////////////////////////// 
		
		mytimer = clock();
		
		cudaPrintfInit(); // set up cuPrintf
                cudaDeviceSynchronize(); 
		
		MH_constant_kernel<<<N_BLOCKS,BLOCK_SIZE>>>(N, dev_spec, device_mcmc_results);
// 		MH_simple_kernel<<<N_BLOCKS,BLOCK_SIZE>>>(N, dev_spec, device_mcmc_results);
// 		MH_detect_local_maxima<<<N_BLOCKS,BLOCK_SIZE>>>(N, dev_spec, device_mcmc_results);
// 		MH_adaptive_temperature_kernel<<<N_BLOCKS,BLOCK_SIZE>>>(N, dev_spec, device_mcmc_results);
// 		MH_prior_kernel<<<N_BLOCKS,BLOCK_SIZE>>>(N, dev_spec, device_mcmc_results);
		cudaDeviceSynchronize(); // wait for preceedings requests to finish
		
		cudaPrintfDisplay(stdout, true); // clean up cuPrintf
		cudaPrintfEnd(); //
		
		cudaError_t error = cudaGetLastError();
		if(error != cudaSuccess) {
			printf("CUDA error: %s\n", cudaGetErrorString(error));
		}
		
		secDEVICE = double(clock() - mytimer) / CLOCKS_PER_SEC;
		
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
			dump_to_file(SAMPLE_PATH.c_str(), host_hypotheses, 0, outer, N, 1); // dump samples
			dump_to_file(MAP_PATH.c_str(),    host_out_MAPs,   0, outer, N, 1); // dump maps
		}

		// make sure our check bits have not changed -- that we didn't overrun anything
		for(int rank=0; rank<N; rank++){
			assert(host_hypotheses[rank].check0 == CHECK_BIT);
			assert(host_hypotheses[rank].check1 == CHECK_BIT);
			assert(host_hypotheses[rank].check2 == CHECK_BIT);
			assert(host_hypotheses[rank].check3 == CHECK_BIT);
			assert(host_hypotheses[rank].check4 == CHECK_BIT);
			assert(host_hypotheses[rank].check5 == CHECK_BIT);
			assert(host_hypotheses[rank].check6 == CHECK_BIT);
		}
		
		delete[] host_hypotheses;
		delete[] host_out_MAPs;
		
		// Now how to handle the end of a "block" -- what update do we do?
		secHOST = double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		// -----------------------------------------------------------------------------------------------------
		// output some performance stats (we do this for burn blocks)
		
		double secTOTAL = secHOST + secTRANSFER + secDEVICE;
		
		unsigned long total_primitives = 0; // count up *approximately* how many primitives were evaluated
		for(int i=0;i<N;i++) total_primitives += host_mcmc_results[i].sample.program_length;
		
		double AR_mean=0.0; // get the accept/reject mean
		for(int i=0;i<N;i++) 
			AR_mean += float(host_mcmc_results[i].acceptance_count) / float(host_mcmc_results[i].proposal_count);
		AR_mean /= float(N);
		
		FILE* fp = fopen(PERFORMANCE_PATH.c_str(), "a");
		fprintf(fp, "%i\t%i\t%.2f\t%.2f\t%.6f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f\t%.5f\n",
			0, 
			outer, 
			PERFECT_LL,
			secDEVICE, 
			secTRANSFER, 
			secHOST, 
			double(N)*double(MCMC_ITERATIONS)/ secTOTAL,
			double(N)*double(MCMC_ITERATIONS)*double(DATA_LENGTH)/secTOTAL,
			double(MCMC_ITERATIONS)*double(DATA_LENGTH)*double(total_primitives)/secTOTAL, 
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
		if( oi==NEG_ && oip1==NEG_) return 0; // no double negatives		
		if( (oi==ONE_) && (oip1==MUL_ || oip1==DIV_ || oip1==LOG_ || oip1==POW_) ) return 0; // none of these operations make sense
		if( (oi==ZERO_) && (oip1==MUL_ || oip1==DIV_ || oip1==ADD_ || oip1==SUB_ || oip1==LOG_ || oip1==EXP_ || oip1==POW_) ) return 0; // none of these operations make sense
		
		if(h->program[i] == X_) found_X = 1;
		if(h->program[i] == CONSTANT_) found_CONSTANT = 1;
		
		
		
	}
	
	return ( found_X && found_CONSTANT);
}

int enum_hyp_i=0;
// int add_hypothesis( hypothesis* h ) {
// 	if( enum_hyp_i >= N) return 0; // can't keep going!
// 		
// 	if(check_in_enumeration(h)) {
// 		
// 	// else add:		
// 	memcpy( (void*)&(host_mcmc_results[enum_hyp_i].sample), (void*)h, sizeof(hypothesis) );
// 	memcpy( (void*)&(host_mcmc_results[enum_hyp_i].MAP),    (void*)h, sizeof(hypothesis) );
// 	enum_hyp_i++;
// 	
// 	return 1;
// 	}
// 	else return 0;
// }



int run_all_hypotheses( hypothesis* h ) {
	
	if(! check_in_enumeration(h) ) return 1; //  don't use, but keep going
	
	// else add:		
	memcpy( (void*)&(host_mcmc_results[enum_hyp_i].sample), (void*)h, sizeof(hypothesis) );
	memcpy( (void*)&(host_mcmc_results[enum_hyp_i].MAP),    (void*)h, sizeof(hypothesis) );
	enum_hyp_i++;

	if( enum_hyp_i >= N) {
		assert(enum_hyp_i == N);
		
		// And run one
		host_run_MCMC(N, host_spec, host_mcmc_results, host_data);
		
		
		// and reset
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
	fprintf(fp, "\tSEED: %i\n", seed);
	fprintf(fp, "\tENUMERATION_DEPTH: %i", ENUMERATION_DEPTH);
	fprintf(fp, "\tMax program length: %i\n", MAX_PROGRAM_LENGTH);
	fprintf(fp, "\tX to constant proportion: %f\n", PRIOR_XtoCONSTANT);
	
	fprintf(fp, "\n\n");
	fclose(fp);
	
	fp = fopen(PERFORMANCE_PATH.c_str(), "w");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << PERFORMANCE_PATH <<"\n"; exit(1);}
	fprintf(fp, "repetition\tblock\tperfect.ll\tdevice.time\ttransfer.time\thost.time\tsamples.per.second\tf.per.second\tprimitives.per.second\ttransfer.mb.per.second\tacceptance.ratio\ttop.overlap.pct\tduplicate.pct\n");
	fclose(fp);
	
	// -----------------------------------------------------------------------
	// Set up the prior
	// -----------------------------------------------------------------------
	
	// Define the prior (above)
	define_prior();

	// Echo the prior
	fp = fopen(LOG_PATH.c_str(), "a");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << LOG_PATH <<"\n"; exit(1);}
	
	fprintf(fp, "\n-----------------------------------------------------------------\n");
	fprintf(fp, "-- Prior:\n");
	fprintf(fp, "-----------------------------------------------------------------\n");
	for(int i=0;i<NUM_OPS;i++) 
		fprintf(fp, "\t%i\t%s\t%f\n", i, NAMES[i], hPRIOR[i]);
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
	// Define a blank hypothesis to initialize everything with
	// -----------------------------------------------------------------------
	
	hypothesis blankhyp;
	blankhyp.program_length=0;
	blankhyp.posterior = -1./0.;
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) { blankhyp.program[i] = NOOP_; }
	for(int i=0;i<MAX_CONSTANTS;i++) { 
		blankhyp.constants[i] = 0.0;
		blankhyp.constant_types[i] = GAUSSIAN;
		
	}
	blankhyp.check0 = CHECK_BIT; blankhyp.check1 = CHECK_BIT; blankhyp.check2 = CHECK_BIT; blankhyp.check3 = CHECK_BIT; blankhyp.check4 = CHECK_BIT; blankhyp.check5 = CHECK_BIT; blankhyp.check6 = CHECK_BIT;
	
	// -----------------------------------------------------------------------
	// Set up the results and specifications locally
	// -----------------------------------------------------------------------
		
	host_mcmc_results = new mcmc_results[N]; // this holds the hypothese we start and stop with
	host_spec = new mcmc_specification[N]; 

	// and copy over
	for(int i=0;i<N;i++){
		COPY_HYPOTHESIS( &(host_mcmc_results[i].sample), &blankhyp);
	}
	
	// -----------------------------------------------------------------------
	// Enumerate the hypotheses
	// -----------------------------------------------------------------------
	
	// Set the specifications
	for(int i=0;i<N;i++) {
		host_spec[i].prior_temperature = 1.0;
		host_spec[i].likelihood_temperature = 1.0;
		host_spec[i].acceptance_temperature = ACCEPTANCE_TEMPERATURE;
		host_spec[i].iterations = MCMC_ITERATIONS; // how many steps to run?
		host_spec[i].initialize = 0;
	}
		
	// To run successively with enumeration (for larger enumeration)
	// This uses host_mcmc_results to store the mcmc results, etc. 
	enumerate_all_programs( N, ENUMERATION_DEPTH, run_all_hypotheses);
	host_run_MCMC(N, host_spec, host_mcmc_results, host_data); // then we still have to run the partially-filled one!
	
	
	/// To enumerate a fixed set and run:
// 	enumerate_all_programs( N, ENUMERATION_DEPTH, add_hypothesis);
// 	host_run_MCMC(N, host_spec, host_mcmc_results, host_data);
	
	// -----------------------------------------------------------------------
	// Cleanup
	// -----------------------------------------------------------------------
	
	delete[] host_data;
	delete[] host_mcmc_results;
	delete[] host_spec;
}
