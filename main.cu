/*
 * GPUropolis - 2017 March 10 - Steve Piantadosi 
 *
 * Simple tree-regeneration on CUDA with coalesced memory access
 * 
 * TOOD: Try copying in the kernel into shared memory
 *       Can always scale x and output
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>
#include <getopt.h>
#include <string.h>
#include <vector>
#include "math_constants.h" // Needed to define CUDART_NAN_F

using namespace std;

const float PRIOR_MULTIPLIER = 10.0; 
const float CONST_PRIOR = 1.0; // how much prior do we pay per constant used?

const int PROGRAM_LENGTH = 15;
const int NCONSTANTS     = 16;

const int CONSTANT_SCALE = 10.0; // Maybe set to be the SD of the y values, fucntions as a scale over the constants in teh prior, proprosals


// Setup for gpu hardware 
const int BLOCK_SIZE = 128;
int N_BLOCKS = 0; // set below
const int HARDARE_MAX_X_BLOCKS = 1024;
const int HARDWARE_MAX_THREADS_PER_BLOCK = 1024; // cannot exceed this many threads per block! For compute level 2.x and greater!


typedef int op; // what type is the program primitives?

typedef struct datum {
    float x;
    float y;
    float sd; // stdev of the output|input. 
} datum;

enum OPS { ZERO, ONE, X, A, B, PLUS, MINUS, RMINUS, DIV, RDIV, LOG, EXP, POW, SQRT, SIN, ASIN, ATAN, NOPS};
const int SQR = -99;


// -----------------------------------------------------------------------
// Memory Macros
// -----------------------------------------------------------------------     

#define CUDA_CHECK() { \
    do {\
        cudaError_t err = cudaGetLastError(); \
        if(err != cudaSuccess) \
            printf("CUDA error: %s\n", cudaGetErrorString(err));\
        } while(0);\
    }
    
// Macro for defining arrays with name device_name, and then copying name over
#define DEVARRAY(type, nom, size) \
    type* device_ ## nom;\
    cudaMalloc((void **) &device_ ## nom, size*sizeof(type)); \
    cudaMemcpy(device_ ## nom, host_ ## nom, size*sizeof(type), cudaMemcpyHostToDevice);\
    CUDA_CHECK()
    
// -----------------------------------------------------------------------
// Numerics
// -----------------------------------------------------------------------     

#define is_valid(x) (isfinite(x) && (x!=CUDART_NAN_F))
#define is_invalid(x) (!is_valid(x))

#define PIf 3.141592653589 
#define BAD_LOGP (-9e99)
 
#define RNG_DEF int& rx
#define RNG_ARGS rx

#define MY_RAND_MAX ((1U << 31) - 1)

// rand call on CUDA
__device__ __host__ int cuda_rand(RNG_DEF) {
   //http://rosettacode.org/wiki/Linear_congruential_generator#C
   return rx = (rx * 1103515245 + 12345) & MY_RAND_MAX; 
}

__device__ __host__ int random_int(int n, RNG_DEF) {
     // number in [0,(n-1)]
    int divisor = MY_RAND_MAX/(n+1);
    int retval;

    do { 
        retval = cuda_rand(RNG_ARGS) / divisor;
    } while (retval >= n);

    return retval;
}

__device__ __host__ float random_float(RNG_DEF) {
    return float(random_int(1000000, RNG_ARGS)) / 1000000.0;
}
// Generate a 0-mean random deviate with mean 0, sd 1
// This uses Box-muller so we don't need branching
__device__ float random_normal(RNG_DEF) {
    float u = random_float(RNG_ARGS);
    float v = random_float(RNG_ARGS);
    return sqrtf(-2.0*logf(u)) * sinf(2*PIf*v);
}

float random_normal() {
    float u = float(rand())/RAND_MAX;
    float v = float(rand())/RAND_MAX;
    return sqrtf(-2.0*log(u)) * sin(2*PIf*v);
}

__device__ float random_cauchy(RNG_DEF) {
    float u = random_normal(RNG_ARGS);
    float v = random_normal(RNG_ARGS);
    return u/v;
}
   
float random_cauchy() {
    float u = random_normal();
    float v = random_normal();
    return u/v;
}
   
__device__ __host__ float lnormalpdf( float x, float s ){
    return -(x*x)/(2.0*s*s) - 0.5 * (logf(2.0*PIf) + 2.0*logf(s));
}

__device__ __host__ float lcauchypdf( float x, float s ){
    return -logf(PIf) - logf(s) - logf(1.0+x*x/s);
}

int rand_lim(int limit) {
//  from http://stackoverflow.com/questions/2999075/generate-a-random-number-within-range/2999130#2999130
//   return a random number between 0 and limit inclusive.
 
    int divisor = RAND_MAX/(limit+1);
    int retval;

    do { 
        retval = rand() / divisor;
    } while (retval > limit);

    return retval;
}

// -----------------------------------------------------------------------
// Loading data
// -----------------------------------------------------------------------     
    
   
// Load data froma  file, putting it into our structs.
// This allows us to trim our data if we want
vector<datum>* load_data_file(const char* datapath, int FIRST_HALF_DATA, int EVEN_HALF_DATA) {

	FILE* fp = fopen(datapath, "r");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << datapath <<"\n"; exit(1);}
	
	vector<datum>* d = new vector<datum>();
	char* line = NULL; size_t len=0; float x,y,sd; 
	while( getline(&line, &len, fp) != -1) {
		if( line[0] == '#' ) continue;  // skip comments
		else if (sscanf(line, "%f\t%f\t%f\n", &x, &y, &sd) == 3) { // floats
			d->push_back( (datum){.x=(float)x, .y=(float)y, .sd=(float)sd} );
		}
		else if (sscanf(line, "%e\t%e\t%e\n", &x, &y, &sd) == 3) { // scientific notation
			d->push_back( (datum){.x=(float)x, .y=(float)y, .sd=(float)sd} );
		}
		else if ( strspn(line, " \r\n\t") == strlen(line) ) { // skip whitespace
			continue;
		}
		else {
			cerr << "*** ERROR IN PARSING INPUT\t" << line << endl;
			exit(1);
		}
	}
	fclose(fp);
	
	// Trim the data based on first/second half or even odd
	if(FIRST_HALF_DATA){
		int mid = d->size()/2;
		for(int i=d->size()-1;i>=mid;i--) {
			d->erase(d->begin()+i);
		}
	}
	if(EVEN_HALF_DATA) {
		for(int i=d->size()-(1-d->size()%2)-1;i>=0;i-=2) {
			d->erase(d->begin()+i);
		}
	}
		
	return d;
}


// -----------------------------------------------------------------------
// Running programs
// -----------------------------------------------------------------------     
    
   
// TODO: Make inline
// evaluat a single operation op on arguments a and b
__device__ float dispatch(op o, float x, float a, float b) {
        switch(o) {
            case ZERO:   return 0.0;
            case ONE:    return 1.0;
            case X:      return x;
            case A:      return a; 
            case B:      return b; // need both since of how consts are passed in
            case PLUS:   return a+b;
            case MINUS:  return a-b;
            case RMINUS: return b-a;
            case DIV:    return a/b;
            case RDIV:   return b/a;
            case SQRT:   return sqrtf(a);
            case SQR:    return a*a;
            case LOG:    return logf(a);
            case SIN:    return sin(a);
            case ASIN:   return asinf(a);
            case ATAN:   return atanf(a/b);
            case EXP:    return expf(a);
            case POW:    return powf(a,b);
            default:     return CUDART_NAN_F;
        }    
}

__device__ float call(int N, int idx, op* P, float* C, float x) {
    float f0 = dispatch(P[idx+0*N], x, C[idx+0*N], C[idx+1*N]);
    float f1 = dispatch(P[idx+1*N], x, C[idx+2*N], C[idx+3*N]);
    float f2 = dispatch(P[idx+2*N], x, f0, f1);
    
    float f3 = dispatch(P[idx+3*N], x, C[idx+4*N], C[idx+5*N]);
    float f4 = dispatch(P[idx+4*N], x, C[idx+6*N], C[idx+7*N]);
    float f5 = dispatch(P[idx+5*N], x, f3, f4);
    
    float f6 = dispatch(P[idx+6*N], x, f2, f5);
    
    
    float f7 = dispatch(P[idx+7*N], x, C[idx+8*N], C[idx+9*N]);
    float f8 = dispatch(P[idx+8*N], x, C[idx+10*N], C[idx+11*N]);
    float f9 = dispatch(P[idx+9*N], x, f7, f8);
    
    float f10 = dispatch(P[idx+10*N], x, C[idx+12*N], C[idx+13*N]);
    float f11 = dispatch(P[idx+11*N], x, C[idx+14*N], C[idx+15*N]);
    float f12 = dispatch(P[idx+12*N], x, f10, f11);
    
    float f13 = dispatch(P[idx+13*N], x, f9, f12);
    
    return dispatch(P[idx+14*N], x, f6, f13);
}

__device__ float compute_likelihood(int N, int idx, op* P, float* C, datum* D, int ndata) {
    float ll = 0.0; // total ll 
    for(int di=0;di<ndata;di++) {
        float fx = call(N, idx, P, C, D[di].x);
	
	if(is_invalid(fx)) return BAD_LOGP;
	  
        ll += lnormalpdf(fx-D[di].y, D[di].sd);
    }
    return ll;
    
}

__device__ float prior_dispatch(op o, float a, float b) {
    // count up the length for the prior
        switch(o) {
            case ZERO: return 0.0;
            case ONE:  return 0.0;
            case X:    return 0.0;
            case A:    return a; 
            case B:    return b; // need both since of how consts are passed in
            case PLUS: return 1+a+b;
            case MINUS: return 1+a+b;
            case RMINUS: return 1+b+a;
            case DIV:   return 1+a+b;
            case RDIV:   return 1+a+b;
            case SQRT:  return 1+a;
            case SQR:   return 1+a;
            case LOG:   return 1+a;
            case SIN:   return 1+a;
            case ASIN:   return 1+a;
            case ATAN:   return 1+a+b;
            case EXP:   return 1+a;
            case POW:   return 1+a+b;
            default:    return CUDART_NAN_F;
        }    
}

__device__ float compute_prior(int N, int idx, op* P, float* C) {
    // compute the prior
    
    int l0 = prior_dispatch(P[idx+0*N], CONST_PRIOR, CONST_PRIOR);
    int l1 = prior_dispatch(P[idx+1*N], CONST_PRIOR, CONST_PRIOR);
    int l2 = prior_dispatch(P[idx+2*N], l0, l1);
    
    int l3 = prior_dispatch(P[idx+3*N], CONST_PRIOR, CONST_PRIOR);
    int l4 = prior_dispatch(P[idx+4*N], CONST_PRIOR, CONST_PRIOR);
    int l5 = prior_dispatch(P[idx+5*N], l3, l4);
    
    int l6 = prior_dispatch(P[idx+6*N], l2, l5);
    
    
    int l7 = prior_dispatch(P[idx+7*N], CONST_PRIOR, CONST_PRIOR);
    int l8 = prior_dispatch(P[idx+8*N], CONST_PRIOR, CONST_PRIOR);
    int l9 = prior_dispatch(P[idx+9*N], l7, l8);
    
    int l10 = prior_dispatch(P[idx+10*N], CONST_PRIOR, CONST_PRIOR);
    int l11 = prior_dispatch(P[idx+11*N], CONST_PRIOR, CONST_PRIOR);
    int l12 = prior_dispatch(P[idx+12*N], l10, l11);
    
    int l13 = prior_dispatch(P[idx+13*N], l9, l12);
    
    int len = prior_dispatch(P[idx+14*N], l6, l13);
    
    float prior = -PRIOR_MULTIPLIER * len;

    for(int c=0;c<NCONSTANTS;c++) {
        prior += lcauchypdf(C[idx+c*N], CONSTANT_SCALE); // proportional to cauchy density
    }
    
    return prior;
    
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MCMC Kernel
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__global__ void MH_simple_kernel(int N, op* P, float* C, datum* D, int ndata, int steps, float* prior, float* likelihood, int random_seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) { return; }  // MUST have this or else all hell breaks loose

    float current_prior = compute_prior(N, idx, P, C);
    float current_likelihood = compute_likelihood(N, idx, P, C, D, ndata);
    float current = current_prior + current_likelihood;
    
    // Two possibilities here. We could let everything do the same proposals (for all steps) in which case we don't
    // add idx. This makes them access the same memory, etc. Alternatively we could add idx and make them separate
    // doesn't seem to matter much for speed
    // int rx = random_seed;     
	int rx = random_seed + idx; // NOTE: We might want to call cuda_rand here once so that we don't have a bias from low seeds
    
	for(int mcmci=0;mcmci<steps;mcmci++) {
        
	    if(mcmci & 0x1 == 0x1) { // propose to a structure every this often
                
                int i = idx + N*random_int(PROGRAM_LENGTH, RNG_ARGS);
                op old = P[i];
                P[i] = random_int(NOPS, RNG_ARGS);
                
                float proposal_prior = compute_prior(N, idx, P, C);
                float proposal_likelihood = compute_likelihood(N, idx, P, C, D, ndata);
                float proposal = proposal_prior + proposal_likelihood;
                        
                if((is_valid(proposal) && (proposal>current || random_float(RNG_ARGS) < expf(proposal-current))) || is_invalid(current)) {
                    current = proposal; // store the updated posterior
                    current_likelihood = proposal_likelihood;
                    current_prior = proposal_prior;
                } else {
                    P[i] = old; // restore the old version
                }

		
	    } else { // propose to a constant otherwise
                
                int i = idx + N*random_int(NCONSTANTS, RNG_ARGS);
                float old = C[i];
    //              C[i] = C[i] + random_normal(RNG_ARGS); 
                C[i] = C[i] + CONSTANT_SCALE*random_cauchy(RNG_ARGS); 
    //             if(random_int(2,RNG_ARGS) == 1)
    //                 C[i] = C[i] * (1.0+0.1*random_normal(RNG_ARGS)); 
    //             else
    //                 C[i] = C[i] / (1.0+0.1*random_normal(RNG_ARGS)); 
                
                float proposal_prior = compute_prior(N, idx, P, C);
                float proposal_likelihood = compute_likelihood(N, idx, P, C, D, ndata);
                float proposal = proposal_prior + proposal_likelihood;
                        
                if((is_valid(proposal) && (proposal>current || random_float(RNG_ARGS) < expf(proposal-current))) || is_invalid(current)) {
                    current = proposal; // store the updated posterior
                    current_likelihood = proposal_likelihood;
                    current_prior = proposal_prior;
                } else {
                    C[i] = old; // restore the old version
                }
                
		
	    }
          
      } // end mcmc loop
	
    prior[idx] = current_prior; 
    likelihood[idx] = current_likelihood;
}




// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Output hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void string_dispatch( char* target, op o, const char* a, const char* b) {
        switch(o) {
            case ZERO:  strcat(target, "0"); break;
            case ONE:   strcat(target, "1");break;
            case X:     strcat(target, "x");break;
            case A:     strcat(target, a); break;
            case B:     strcat(target, b); break;// need both since of how consts are passed in
            case PLUS:  strcat(target, "("); strcat(target, a); strcat(target, "+"); strcat(target, b); strcat(target, ")"); break;
            case MINUS: strcat(target, "("); strcat(target, a); strcat(target, "-"); strcat(target, b); strcat(target, ")"); break;
            case RMINUS:strcat(target, "("); strcat(target, b); strcat(target, "-"); strcat(target, a); strcat(target, ")"); break;
            case DIV:   strcat(target, "("); strcat(target, a); strcat(target, "/"); strcat(target, b); strcat(target, ")"); break;
            case RDIV:  strcat(target, "("); strcat(target, b); strcat(target, "/"); strcat(target, a); strcat(target, ")"); break;
            case LOG:   strcat(target, "log("); strcat(target, a); strcat(target, ")"); break;
            case SIN:   strcat(target, "sin("); strcat(target, a); strcat(target, ")"); break;
            case ASIN:  strcat(target, "asin("); strcat(target, a); strcat(target, ")"); break;
            case ATAN:  strcat(target, "atan("); strcat(target, a); strcat(target, "/"); strcat(target, b); strcat(target, ")"); break;
            case SQR:   strcat(target, "(("); strcat(target, a); strcat(target, ")^2)"); break;
            case SQRT:  strcat(target, "sqrt("); strcat(target, a); strcat(target, ")"); break;
            case EXP:   strcat(target, "exp("); strcat(target, a); strcat(target, ")"); break;
            case POW:   strcat(target, "("); strcat(target, a); strcat(target, "^"); strcat(target, b); strcat(target, ")"); break;
            default:    strcat(target, "NAN"); break;
        }    
}

char buf[PROGRAM_LENGTH][1000]; 
char cbuf[100]; // for consts
char cbuf2[100]; // for consts
void displaystring(const char* const_format, int N, int idx, op* P, float* C, FILE* fp){
    // pass in the constant format so we can give "C" if we want to get the structure out
    
    for(int i=0;i<PROGRAM_LENGTH;i++) { strcpy(buf[i], ""); }
    
    sprintf(cbuf, const_format, C[idx+0*N]); sprintf(cbuf2, const_format, C[idx+1*N]);
    string_dispatch(buf[0], P[idx+0*N], cbuf, cbuf2);
    sprintf(cbuf, const_format, C[idx+2*N]); sprintf(cbuf2, const_format, C[idx+3*N]);
    string_dispatch(buf[1], P[idx+1*N], cbuf, cbuf2);
    string_dispatch(buf[2], P[idx+2*N], buf[0], buf[1]);

    sprintf(cbuf, const_format, C[idx+4*N]); sprintf(cbuf2, const_format, C[idx+5*N]);
    string_dispatch(buf[3], P[idx+3*N], cbuf, cbuf2);
    sprintf(cbuf, const_format, C[idx+6*N]); sprintf(cbuf2, const_format, C[idx+7*N]);
    string_dispatch(buf[4], P[idx+4*N], cbuf, cbuf2);
    string_dispatch(buf[5], P[idx+5*N], buf[3], buf[4]);

    string_dispatch(buf[6], P[idx+6*N], buf[2], buf[5]);


    sprintf(cbuf, const_format, C[idx+8*N]); sprintf(cbuf2, const_format, C[idx+9*N]);
    string_dispatch(buf[7], P[idx+7*N], cbuf, cbuf2);
    sprintf(cbuf, const_format, C[idx+10*N]); sprintf(cbuf2, const_format, C[idx+11*N]);
    string_dispatch(buf[8], P[idx+8*N], cbuf, cbuf2);
    string_dispatch(buf[9], P[idx+9*N], buf[7], buf[8]);

    sprintf(cbuf, const_format, C[idx+12*N]); sprintf(cbuf2, const_format, C[idx+13*N]);
    string_dispatch(buf[10], P[idx+10*N], cbuf, cbuf2);
    sprintf(cbuf, const_format, C[idx+14*N]); sprintf(cbuf2, const_format, C[idx+15*N]);
    string_dispatch(buf[11], P[idx+11*N], cbuf, cbuf2);
    string_dispatch(buf[12], P[idx+12*N], buf[10], buf[11]);

    string_dispatch(buf[13], P[idx+13*N], buf[9], buf[12]);
    
    string_dispatch(buf[14], P[idx+14*N], buf[6], buf[13]);
    
    fprintf(fp, "%s", buf[14]);
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
        {"steps",   required_argument,    NULL, 'i'},
        {"N",            required_argument,    NULL, 'N'},
        {"out",          required_argument,    NULL, 'O'},
        {"outer",        required_argument,    NULL, 'o'},
        {"seed",         required_argument,    NULL, 's'},
        {"burn",         required_argument,    NULL, 'b'},
        {"gpu",          required_argument,    NULL, 'g'},
        {"first-half",   no_argument,    NULL, 'f'},
        {"even-half",    no_argument,    NULL, 'e'},
        {"all",    no_argument,    NULL, '_'},
        {NULL, 0, 0, 0} // zero row for bad arguments
    };  

int main(int argc, char** argv)
{   
    
    /// default parameters
    int N = 10000;
    int steps = 1000;
    int outer = 100;
    int seed = -1;
//     int burn = 10;
    int WHICH_GPU = 0;
    int FIRST_HALF_DATA = 0;
    int EVEN_HALF_DATA = 0;
    float resample_threshold = 500.0; // hypotheses more than this far from the MAP get resampled every outer block
    string in_file_path = "data.txt";
    string out_path = "out/";
    
    // ----------------------------------------------------------------------------
    // Parse command line
    // -----------------------------------------------------------------------
    
    int option_index = 0, opt=0;
    while( (opt = getopt_long( argc, argv, "bp", long_options, &option_index )) != -1 )
        switch( opt ) {
            case 'd': in_file_path = optarg; break;
            case 'i': steps = atoi(optarg); break;
            case 'N': N = atoi(optarg); break;
            case 'o': outer = atoi(optarg); break;
            case 'O': out_path = optarg; break;
//             case 'b': BURN_BLOCKS = atoi(optarg); break;
            case 'g': WHICH_GPU = atoi(optarg); break;
            case 's': seed = (float)atof(optarg); break;
            case 'f': FIRST_HALF_DATA = 1; break;
            case 'e': EVEN_HALF_DATA = 1; break;
            case '_': break; // don't do anything if we use all the data
            
            default: return 1; // unspecified
        }
    
    // -----------------------------------------------------------------------
    // Initialize the GPU
    // -----------------------------------------------------------------------
     
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if(WHICH_GPU <= deviceCount) {
        cudaSetDevice(WHICH_GPU);
    }
    else {
        cerr << "Invalid GPU device " << WHICH_GPU << endl;
        return 1;
    }
    
    // -----------------------------------------------------------------------
    // Set up the blocks
    // -----------------------------------------------------------------------
        
    int N_BLOCKS = N/BLOCK_SIZE + (N%BLOCK_SIZE == 0 ? 0:1);
    
    assert(N_BLOCKS < HARDARE_MAX_X_BLOCKS); // can have at most this many blocks
    assert(N/N_BLOCKS <= HARDWARE_MAX_THREADS_PER_BLOCK); // MUST HAVE LESS THREADS PER BLOCK!!
    
    // -----------------------------------------------------------------------
    // Set up the output files etc
    // -----------------------------------------------------------------------
    
    string SAMPLE_PATH = out_path+"/samples.txt";
    string LOG_PATH = out_path+"/log.txt";
//     string PERFORMANCE_PATH = out_path+"/performance.txt";
    
    // -------------------------------------------------------------------------
    // Make the RNG replicable
    
    if(seed==-1) { srand(time(NULL)); seed = rand(); }

    // -------------------------------------------------------------------------
    // Write the log and performance log
    
    FILE* fp = fopen(LOG_PATH.c_str(), "w");
    if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << LOG_PATH <<"\n"; exit(1);}
    
    fprintf(fp, "-----------------------------------------------------------------\n");
    fprintf(fp, "-- Parameters:\n");
    fprintf(fp, "-----------------------------------------------------------------\n");
    fprintf(fp, "\tInput data path: %s\n", in_file_path.c_str());
    fprintf(fp, "\tOutput path: %s\n", out_path.c_str());
    fprintf(fp, "\tMCMC Iterations (per block): %i\n", steps);
    fprintf(fp, "\tBlocks: %i\n", outer);
//     fprintf(fp, "\tBurn Blocks: %i\n", BURN_BLOCKS);
    fprintf(fp, "\tN chains: %i\n", N);
    fprintf(fp, "\tseed: %i\n", seed);
    fprintf(fp, "\tMax program length: %i\n", PROGRAM_LENGTH);
    fprintf(fp, "\tN Constants: %i\n", NCONSTANTS);
    fprintf(fp, "\n\n");
    fclose(fp);
    
    // -----------------------------------------------------------------------
    // Read the data and set up some arrays
    // -----------------------------------------------------------------------
    
    vector<datum>* data_vec = load_data_file(in_file_path.c_str(), FIRST_HALF_DATA, EVEN_HALF_DATA);
    datum* host_D = &((*data_vec)[0]); // can do this with vectors now
    int ndata = data_vec->size();
    
    
    // Echo the data we actually run with (post-filtering for even/firsthalf)
    fp = fopen(LOG_PATH.c_str(), "a");
    fprintf(fp, "\n-----------------------------------------------------------------\n");
    fprintf(fp, "-- Data:\n");
    fprintf(fp, "-----------------------------------------------------------------\n");
    for(int i=0;i<ndata;i++) 
        fprintf(fp, "\t%f\t%f\t%f\n", host_D[i].x, host_D[i].y, host_D[i].sd);
    fclose(fp);
    
    // -----------------------------------------------------------------------
    // Set up all the programs locally
    // -----------------------------------------------------------------------
    
    op*    host_P =   new op[N*PROGRAM_LENGTH];
    float* host_C =  new float[N*NCONSTANTS];
    float* host_prior =  new float[N];
    float* host_likelihood =  new float[N]; 
    
   
    for(int i=0;i<PROGRAM_LENGTH*N;i++) host_P[i] = rand_lim(NOPS-1);
    for(int i=0;i<NCONSTANTS*N;i++)     host_C[i] = CONSTANT_SCALE*random_cauchy();
    
    
    // -----------------------------------------------------------------------
    // Allocate on device and copy
    // -----------------------------------------------------------------------


    DEVARRAY(datum, D, ndata) // defines device_D, 
    DEVARRAY(op,    P, N*PROGRAM_LENGTH) // device_P
    DEVARRAY(float, C, N*NCONSTANTS) // device_C
    DEVARRAY(float, prior, N) 
    DEVARRAY(float, likelihood, N) 
    
    
    // -----------------------------------------------------------------------
    // Run
    // -----------------------------------------------------------------------
    
    // Set the specifications
    for(int o=0;o<outer;o++) {
        
        // run this many steps
        // interesting, without the rand() call, we have eventual decreases in posterior probability, 
        MH_simple_kernel<<<N_BLOCKS,BLOCK_SIZE>>>(N, device_P, device_C, device_D, ndata, steps, device_prior, device_likelihood, rand());
    
        // copy memory back 
        cudaMemcpy(host_P, device_P, N*sizeof(op)*PROGRAM_LENGTH, cudaMemcpyDeviceToHost); CUDA_CHECK();
        cudaMemcpy(host_C, device_C, N*sizeof(float)*NCONSTANTS, cudaMemcpyDeviceToHost); CUDA_CHECK();
        cudaMemcpy(host_prior, device_prior, N*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK();
        cudaMemcpy(host_likelihood, device_likelihood, N*sizeof(float), cudaMemcpyDeviceToHost); CUDA_CHECK();
        cudaDeviceSynchronize(); // wait for preceedings requests to finish
        
        // and now print
        fp = fopen(SAMPLE_PATH.c_str(), "a");
        for(int h=0;h<N;h++) {
	    fprintf(fp, "%d\t%d\t", h, o);
	    
            fprintf(fp, "%.4f\t%4.f\t%.4f\t", host_prior[h]+host_likelihood[h], host_prior[h], host_likelihood[h]);
             
//             for(int c=0;c<PROGRAM_LENGTH;c++){ 
//                 fprintf(fp, "%d ", host_P[h+N*c]);
//             }
//             fprintf(fp, "\t\"");

            fprintf(fp, "\"");
            displaystring("C", N, h, host_P, host_C, fp);
            
            fprintf(fp, "\"\t\"");
            
            displaystring("%.4f", N, h, host_P, host_C, fp);
            fprintf(fp, "\"\t");
            
            for(int c=0;c<NCONSTANTS;c++){ 
                fprintf(fp, "%.2f\t", host_C[h+N*c]);
            }
            
            fprintf(fp, "\n");
            
        }
        fclose(fp);
        
        // find the MAP so far
        float map_so_far = -9e99;
        for(int h=0;h<N;h++) { 
            if(host_prior[h]+host_likelihood[h] > map_so_far) 
                map_so_far = host_prior[h]+host_likelihood[h];
        }
        
        // Now resample from the prior if we are too bad
        for(int h=0;h<N;h++) {
            if(host_prior[h]+host_likelihood[h] < map_so_far - resample_threshold) {
                // randomize
                for(int i=0;i<PROGRAM_LENGTH;i++) host_P[h+i*N] = rand_lim(NOPS-1);
                for(int i=0;i<NCONSTANTS;i++)     host_C[h+i*N] = CONSTANT_SCALE*random_cauchy();
            }
        }
        // and copy back 
        cudaMemcpy(device_P, host_P, N*PROGRAM_LENGTH*sizeof(op), cudaMemcpyHostToDevice);  CUDA_CHECK();
        cudaMemcpy(device_C, host_C, N*NCONSTANTS*sizeof(float), cudaMemcpyHostToDevice);  CUDA_CHECK();
    }
        
        
    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    
    delete[] host_P;
    delete[] host_C;
    delete[] host_prior;
    delete[] host_likelihood;
    delete[] host_D;
    
}
