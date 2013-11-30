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

// A hacked prior to penalize extra deep X, corresponding to very weirdo functions
const float X_DEPTH_PENALTY = 100.0; // extra penalty for X depth. 0 here gives PCFG generation probability prior
const float X_PENALTY = 10.0; // penalty for using X

// Specification of the prior
// in tree resampling, the expected length here is important in getting a good acceptance rate -- too low
// (meaning too long) and we will reject almost everything
const float EXPECTED_LENGTH = 3.0; // also the expected length of proposals
const float PRIOR_XtoCONSTANT = 0.1; //what proportion of constant proposals are x (as opposed to all other constants)?

#include "src/misc.cu"
#include "src/__PRIMITIVES.cu"
#include "src/data.cu"
#include "src/hypothesis.cu"
#include "src/programs.cu"
#include "src/mcmc-specification.cu"
#include "src/mcmc-results.cu"
#include "src/virtual-machine.cu"
#include "src/hypothesis-array.cu"
#include "src/kernels/MH-simple-kernel.cu"

using namespace std;

int N = 1024;  // Hw many chains?
int NTOP = 5000; // store this many of the "top" hypotheses 

const int BLOCK_SIZE = 128; // WOW 16 appears to be fastest here...
int N_BLOCKS = 0; // set below
const int HARDARE_MAX_X_BLOCKS = 1024;
const int HARDWARE_MAX_THREADS_PER_BLOCK = 1024; // cannot exceed this many threads per block! For compute level 2.x and greater!

string in_file_path = "data.txt"; 
string OUT_PATH     = "run";

int SEED = -1; // Random number seed (for replicability) if -1, we use time()

int REPETITONS = 1; // how many outer loops do we repeat?
int MCMC_ITERATIONS = 1000; 
int OUTER_BLOCKS = 1;
int BURN_BLOCKS = 0; // how many blocks (of MCMC_ITERATIONS each) do we burn-in?

int FIRST_HALF_DATA = 0; // use only the first half of the data
int EVEN_HALF_DATA  = 0; // use only the even half of the data

int END_OF_BLOCK_ACTION = 2; // an integer code for 
/* 
 * 1: start anew each outer loop (restart from prior)
 * 2: maintain the same chain (just print the most recent sample)
 */
/*
 * TODO: NOT IMPLEMENTED:
 * 5: resample from the top, penalizing by the number of samples already drawn from that hypothesis. So new things of high rank are 
// double MAIN_RESAMPLE_DISCOUNT = 1.0; // the posterior is penalized by this * [the number of chains started here], so that we will explore newer regions of the space preferentially (even if they are not high probability mass). If this is set to 0.0, then we just resample from the real posterior. If it's +inf, we only restart a chain once

// double RESAMPLE_IF_LOWER = 1000.0; // if we are this much lower than the max, we will be resampled from the top. 
*/

static struct option long_options[] =
	{	
		{"in",           required_argument,    NULL, 'd'},
		{"iterations",   required_argument,    NULL, 'i'},
		{"repetitions",  required_argument,    NULL, 'R'},
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
			case 'R': REPETITONS = atoi(optarg); break;
			case 'b': BURN_BLOCKS = atoi(optarg); break;
			case 's': SEED = (float)atof(optarg); break;
			case 'f': FIRST_HALF_DATA = 1; break;
			case 'm': END_OF_BLOCK_ACTION = atoi(optarg); break;
			case 'e': EVEN_HALF_DATA = 1; break;
			case 'L': set_MAX_PROGRAM_LENGTH(atoi(optarg)); break;
			case '_': break; // don't do anything if we use all the data
			case 'h': // help output:
// 				cout << "Options: " << endl;
// 				cout << "\t--max-base=N         sets the maximum base to N" << endl;
// 				cout << "\t--max-power=N        sets the maximum power to N" << endl;
				return 0;
			default:
				return 1;
		}
	
	N_BLOCKS = N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1);
	
	assert(N_BLOCKS < HARDARE_MAX_X_BLOCKS); // can have at most this many blocks
	assert(N/N_BLOCKS <= HARDWARE_MAX_THREADS_PER_BLOCK); // MUST HAVE LESS THREADS PER BLOCK!!
	
	
	// -----------------------------------------------------------------------
	// Set up the output files etc
	// -----------------------------------------------------------------------
	
	string SAMPLE_PATH = OUT_PATH+"/samples.txt";
	string MAP_PATH = OUT_PATH+"/MAPs.txt";
	string TOP_PATH = OUT_PATH+"/tops.txt";
// 	string SAMPLE_BINARY_PATH = OUTPATH+"/state"; // just a dump of host_hypotheses
	string LOG_PATH = OUT_PATH+"/log.txt";
	string PERFORMANCE_PATH = OUT_PATH+"/performance.txt";
	
	// -------------------------------------------------------------------------
	// Make the RNG replicable
	
	int seed;
	if(SEED==-1) {
		srand(time(NULL));
		seed = rand();
	} 
	else {
		seed = SEED;
	}

	// -------------------------------------------------------------------------
	// Write the log and performance log
	
	FILE* fp = fopen(LOG_PATH.c_str(), "w");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << LOG_PATH <<"\n"; exit(1);}
	
	fprintf(fp, "-----------------------------------------------------------------\n");
	fprintf(fp, "-- Parameters:\n");
	fprintf(fp, "-----------------------------------------------------------------\n");
	fprintf(fp, "\tInput data path: %s\n", in_file_path.c_str());
	fprintf(fp, "\tOutput path: %s\n", OUT_PATH.c_str());
	fprintf(fp, "\tRepetitions: %i\n", REPETITONS);
	fprintf(fp, "\tMCMC Iterations (per block): %i\n", MCMC_ITERATIONS);
	fprintf(fp, "\tBlocks: %i\n", OUTER_BLOCKS);
	fprintf(fp, "\tBurn Blocks: %i\n", BURN_BLOCKS);
	fprintf(fp, "\tN chains: %i\n", N);
	fprintf(fp, "\tSEED: %i\n", seed);
	fprintf(fp, "\tEnd of block action: %i\n", END_OF_BLOCK_ACTION);
	fprintf(fp, "\tMax program length: %i\n", hMAX_PROGRAM_LENGTH);
	fprintf(fp, "\tX to constant proportion: %f\n", PRIOR_XtoCONSTANT);
	
	fprintf(fp, "\n\n");
	fclose(fp);
	
	fp = fopen(PERFORMANCE_PATH.c_str(), "w");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << PERFORMANCE_PATH <<"\n"; exit(1);}
	fprintf(fp, "repetition\tblock\tperfect.ll\tMAP.ll\tdevice.time\ttransfer.time\thost.time\tsamples.per.second\tf.per.second\tprimitives.per.second\ttransfer.mb.per.second\tacceptance.ratio\n");
	fclose(fp);
	
	// -----------------------------------------------------------------------
	// Set up the prior
	// -----------------------------------------------------------------------
	
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
	float P_1arg  = P / float(count_args[1] + count_args[2]);
	float P_2arg  = P / float(count_args[1] + count_args[2]);;
	
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
	for(int i=0;i<NUM_OPS;i++) hPRIOR[i] /= priorZ;

	// and copy PRIOR over to the device
	cudaMemcpyToSymbol(dPRIOR, hPRIOR, MAX_NUM_OPS*sizeof(float), 0, cudaMemcpyHostToDevice);
	
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
	datum* host_data = &((*data_vec)[0]); // can do this with vectors now
	
	const int DLEN = data_vec->size();
	const size_t DATA_BYTE_LEN = DLEN*sizeof(datum);

	// compute the maximum possible ll
	// we use this for the start of annealing temperature
	double PERFECT_LL = 0.0;
	for(int di=0;di<DLEN;di++) {PERFECT_LL += lnormalpdf( 0.0, host_data[di].sd); }

	// and put this on the GPU
	datum* device_data; 
	cudaMalloc((void **) &device_data, DATA_BYTE_LEN);
	cudaMemcpy(device_data, host_data, DATA_BYTE_LEN, cudaMemcpyHostToDevice);
	
	// Echo the data we actually run with (post-filtering for even/firsthalf)
	fp = fopen(LOG_PATH.c_str(), "a");
	fprintf(fp, "\n-----------------------------------------------------------------\n");
	fprintf(fp, "-- Data:\n");
	fprintf(fp, "-----------------------------------------------------------------\n");
	for(int i=0;i<DLEN;i++) 
		fprintf(fp, "\t%f\t%f\t%f\n", host_data[i].input, host_data[i].output, host_data[i].sd);
	fclose(fp);
	
	// -----------------------------------------------------------------------
	// Initialize RNG
	// -----------------------------------------------------------------------
	
	// define random numbers:
	int rx,ry,rz,rw;
	rx = rand(); ry = rand(); rz = rand(); rw = rand();
	
	// this is how we seed each chain
	long rng_seed = 0;
	
	// -----------------------------------------------------------------------
	// Set up some bits...
	// -----------------------------------------------------------------------
	
	// For locally manipulating hypotheses
	size_t HYPOTHESIS_ARRAY_SIZE = N * sizeof(hypothesis);
	hypothesis* host_hypotheses = new hypothesis[N]; 
	hypothesis* host_hypothesis_tmp = new hypothesis[N]; 
	hypothesis* host_out_MAPs = new hypothesis[N];
	
	int MCMC_RESULTS_SIZE = sizeof(mcmc_results)*N;
	mcmc_results* host_mcmc_results = new mcmc_results[N];
	mcmc_results* device_mcmc_results; cudaMalloc((void **) &device_mcmc_results, MCMC_RESULTS_SIZE); // device allocate
	
	// initialize this local array (copied to CUDA after any further changes)
	for(int n=0;n<N;n++){
		initialize(&(host_mcmc_results[n].MAP),    RNG_ARGS);
		initialize(&(host_mcmc_results[n].sample), RNG_ARGS);
	}
	
	// store the top hypotheses overall
	hypothesis* host_top_hypotheses = new hypothesis[NTOP+2*N]; 
	for(int i=0;i<NTOP+2*N;i++) initialize(&host_top_hypotheses[i], RNG_ARGS);
	
	// a special guy we keep  empty
	hypothesis blankhyp; initialize(&blankhyp, RNG_ARGS);
	
	// -----------------------------------------------------------------------
	// Make the specification for mcmc steps, iterations, etc.
	// -----------------------------------------------------------------------
	
	int NSPECSIZE = N*sizeof(mcmc_specification);
	
	mcmc_specification* host_spec = new mcmc_specification[N]; 
	mcmc_specification* dev_spec; cudaMalloc((void **) &dev_spec, NSPECSIZE ); // device allocate

	
	// And set up the specifications!
	for(int i=0;i<N;i++) {
		host_spec[i].prior_temperature = 1.0;
		host_spec[i].likelihood_temperature = 1.0;
		host_spec[i].acceptance_temperature = 1.0;
		host_spec[i].iterations = MCMC_ITERATIONS; // how many steps to run?
		host_spec[i].initialize = 1;
		host_spec[i].rng_seed = 0; // set below
		host_spec[i].data_length = DLEN;
		host_spec[i].data = device_data;
		// TODO: Max program length should be in here! But that's hard, unless that's also a feature of hyp...
	}
	
	
	// -----------------------------------------------------------------------
	// Main loop
	// -----------------------------------------------------------------------
	
	clock_t mytimer;
	
	
	for(int rep=0; rep<REPETITONS; rep++) { // how many repetitions do we do?
	for(int outer=0;outer<OUTER_BLOCKS+BURN_BLOCKS;outer++) {
		double secDEVICE=0.0, secHOST=0.0, secTRANSFER=0.0; // how long do we spend on each?	
		
		// increase the max program length, as a form of tempering...
		// set_MAX_PROGRAM_LENGTH(min(outer,MAX_MAX_PROGRAM_LENGTH));
		
		// Set the specifications
		for(int i=0;i<N;i++) {
			host_spec[i].initialize = (outer==0)||(END_OF_BLOCK_ACTION==1);
			
			
			host_spec[i].rng_seed = seed + rng_seed++; // set this seed
// 			cerr << host_spec[i].initialize << "\t" << host_spec[i].rng_seed << endl;
			// THIS IS ANNEALING ON THE LIKELIHOOD
// 			host_spec[i].likelihood_temperature = 1.0 + 5.0 / float(outer+1.0);
		}
		mytimer = clock();
		cudaMemcpy(dev_spec, host_spec, NSPECSIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(device_mcmc_results, host_mcmc_results, MCMC_RESULTS_SIZE, cudaMemcpyHostToDevice);
		secTRANSFER += double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		//////////////// Now run: //////////////// 
		
		mytimer = clock();
		
		cudaDeviceSynchronize(); 
		MH_simple_kernel<<<N_BLOCKS,BLOCK_SIZE>>>(N, dev_spec, device_mcmc_results);
		cudaDeviceSynchronize(); // wait for preceedings requests to finish
		
		secDEVICE = double(clock() - mytimer) / CLOCKS_PER_SEC;
				
		// Retrieve result from device and store it in host array
		mytimer = clock();
		cudaMemcpy(host_mcmc_results, device_mcmc_results, MCMC_RESULTS_SIZE, cudaMemcpyDeviceToHost);
		secTRANSFER += double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		mytimer = clock(); // for timing the rest of host operations
		
		// and copy the host hypotheses over to hosts and maps
		for(int i=0;i<N;i++){ 
// 			cerr <<  i << "\t" << host_mcmc_results[i].sample.posterior << endl;
			COPY_HYPOTHESIS( &host_hypotheses[i], &(host_mcmc_results[i].sample));
			COPY_HYPOTHESIS( &host_out_MAPs[i],   &(host_mcmc_results[i].MAP));
		}		
		
		// sort them as required below
		qsort( (void*)host_out_MAPs,   N, sizeof(hypothesis), hypothesis_posterior_compare);
		qsort( (void*)host_hypotheses, N, sizeof(hypothesis), hypothesis_posterior_compare);
		
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

		// -----------------------------------------------------------------------------------------------------
		// Save results. But not for burn blocks
		
		// print out
		if(outer >= BURN_BLOCKS) {
			dump_to_file(SAMPLE_PATH.c_str(), host_hypotheses, rep, outer, N, 1); // dump samples
			dump_to_file(MAP_PATH.c_str(), host_out_MAPs, rep, outer, N, 1); // dump maps
		}
		
		// -----------------------------------------------------------------------------------------------------
		// Manage our collection of the top hypotheses (here: samples, not MAPs)
		// we put these in host_top_hypotheses, and the first NTOP of these always stores the best
		// we maintain this by copying all of host_out_MAPs to the end, resorting, and removing duplicates
		
		memcpy( (void*)&host_top_hypotheses[NTOP], (void*)host_out_MAPs, HYPOTHESIS_ARRAY_SIZE); // put these at the end
		memcpy( (void*)&host_top_hypotheses[NTOP+N], (void*)host_hypotheses, HYPOTHESIS_ARRAY_SIZE); // put these at the end
		
		
		// resort, best first, putting duplicate programs next to each other
 		qsort(  (void*)host_top_hypotheses, NTOP+2*N, sizeof(hypothesis), sort_bestfirst_unique); 
		// and now delete duplicates
		delete_duplicates(host_top_hypotheses, NTOP, NTOP+2*N, &blankhyp);
		// and save this
		dump_to_file(TOP_PATH.c_str(), host_top_hypotheses, rep, outer, NTOP, 0);
				
		// -----------------------------------------------------------------------------------------------------
		// Now how to handle the end of a "block" -- what update do we do?
			
		secHOST = double(clock() - mytimer) / CLOCKS_PER_SEC;
		
		// -----------------------------------------------------------------------------------------------------
		// output some performance stats (we do this for burn blocks)
		
		double secTOTAL = secHOST + secTRANSFER + secDEVICE;
		
		unsigned long total_primitives = 0; // count up *approximately* how many primitives were evaluated
		for(int i=0;i<N;i++) total_primitives += host_hypotheses[i].program_length;
		
		double AR_mean=0.0; // get the accept/reject mean
		for(int i=0;i<N;i++) 
			AR_mean += float(host_mcmc_results[i].acceptance_count) / float(host_mcmc_results[i].proposal_count);
		AR_mean /= float(N);
		
		FILE* fp = fopen(PERFORMANCE_PATH.c_str(), "a");
		fprintf(fp, "%i\t%i\t%.2f\t%.2f\t%.6f\t%.6f\t%.6f\t%.2f\t%.2f\t%.2f\t%.2f\t%.5f\n",
			rep, 
			outer, 
			PERFECT_LL,
			host_top_hypotheses[0].likelihood, 
			secDEVICE, 
			secTRANSFER, 
			secHOST, 
			double(N)*double(MCMC_ITERATIONS)/ secTOTAL,
			double(N)*double(MCMC_ITERATIONS)*double(DLEN)/secTOTAL,
			double(MCMC_ITERATIONS)*double(DLEN)*double(total_primitives)/secTOTAL, 
			double(HYPOTHESIS_ARRAY_SIZE*2)/(1048576. * secTRANSFER),
			AR_mean
       		);
		fclose(fp);
	
	} // end outer loop 
	} // end repetitions
	
	// -----------------------------------------------------------------------
	// Cleanup
	// -----------------------------------------------------------------------
	
	delete[] host_hypotheses;
	delete[] host_hypothesis_tmp;
	delete[] host_data;
	delete[] host_mcmc_results;
	delete[] host_spec;
	
	cudaFree(device_data);
	cudaFree(dev_spec);
	cudaFree(device_mcmc_results);
	
}