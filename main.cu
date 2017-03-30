/*
 * GPUropolis - 2017 March 10 - Steve Piantadosi 
 *
 * Simple tree-regeneration on CUDA with coalesced memory access
 * 

    This version uses constants at the bottom of the tree, instead of a constant at each 
    program position that is potentially available to a constant-op 
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
#include "math_constants.h" // Needed to define CUDART_NAN_F, CUDART_INF_F

using namespace std;

const float PRIOR_MULTIPLIER = 1.0; 
const float CONST_LENGTH_PRIOR = 1.0; // how mcuh do constants cost in terms of length?
const float X_LENGTH_PRIOR = 1.0; // how mcuh does X cost in terms of length?

const int PROGRAM_LENGTH = 7;//15;
const int NCONSTANTS     = 7;//15; // these must be equal in this version -- one constant for each program slot; although note the low ones are never used, right?

const float CONSTANT_SCALE = 10.0; // Maybe set to be the SD of the y values, fucntions as a scale over the constants in teh prior, proprosals

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
//          1     I   a  b   +      -       _      #      *      @      /    |     L    E    ^     p    V      P     R     S     A    T      G      B      A
enum OPS { PONE, INV, A, B, PLUS, MINUS, RMINUS, CPLUS, TIMES, CTIMES, DIV, RDIV, LOG, EXP, POW, CPOW, RPOW, CRPOW, SQRT, SIN, ASIN, ATAN, GAMMA, BESSEL, ABS,     NOPS};
const int SQR = -99; // if we want to remove some from OPS, use here so the code below doesn't break
const char* PROGRAM_CODE = "1Iab+-_#*@/|LE^pVPRSATGBA"; // if we want to print a concise description of the program (mainly for debugging) These MUST be algined with OPS


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
    return sqrtf(-2.0*logf(u)) * sinf(2.0*PIf*v);
}

float random_normal() {
    float u = float(rand())/RAND_MAX;
    float v = float(rand())/RAND_MAX;
    return sqrtf(-2.0*log(u)) * sin(2.0*PIf*v);
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Run programs
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   
// TODO: Make inline
// evaluat a single operation op on arguments a and b
__device__ float dispatch(op o, float x, float a, float b, float C) {
        switch(o) {
            case PONE:   return a + 1.0;
            case INV:    return fdividef(1.0,a);
            case A:      return a; 
            case B:      return b; // need both since of how consts are passed in
            case PLUS:   return a+b;
            case CPLUS:  return a+C;
            case MINUS:  return a-b;
            case RMINUS: return b-a;
            case TIMES:  return a*b;
            case CTIMES: return C*a;
            case DIV:    return fdividef(a,b);
            case RDIV:   return fdividef(b,a);
            case SQRT:   return sqrtf(a);
            case SQR:    return a*a;
            case LOG:    return logf(a);
            case SIN:    return sin(a);
            case ASIN:   return asinf(a);
            case ATAN:   return atanf(a);
            case EXP:    return expf(a);
            case POW:    return powf(a,b);
            case CPOW:   return powf(a,C);
            case RPOW:   return powf(b,a);
            case CRPOW:  return powf(C,a);
            case GAMMA:  return tgammaf(a);
            case BESSEL: return j0f(a);
            case ABS:    return fabsf(a);
            default:     return CUDART_NAN_F;
        }    
}

__device__ float call(int N, int idx, op* P, float* C, float x) {
    // start at the first non-leaves
    float buf[PROGRAM_LENGTH+1]; // we'll waste one space here at the beginning to simplify everything else
    
    for(int i=PROGRAM_LENGTH;i>=1;i--) { // start at the last node
        int lidx = 2*i; // indices of the children
        int ridx = 2*i+1;
        
        if(lidx > PROGRAM_LENGTH) buf[i] = x; // the default value at the base of the tree
        else                      buf[i] = dispatch(P[idx+(i-1)*N], x, buf[lidx], buf[ridx], C[idx+(i-1)*N]); // P[...-1] since P is zero-indexed
    }
    // now buf[1] stores the output, which is the top node
    
    return buf[1];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the likelihood
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float compute_likelihood(int N, int idx, op* P, float* C, datum* D, int ndata) {
    float ll = 0.0; // total ll 
    for(int di=0;di<ndata;di++) {
        
        float fx = call(N, idx, P, C, D[di].x);
	if(is_invalid(fx)) return -CUDART_INF_F;
	  
        float l = lnormalpdf(fx-D[di].y, D[di].sd);
        if(is_invalid(l)) return -CUDART_INF_F;
        
        ll += l;
    }
    return ll;
    
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the prior
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ float dispatch_length(op o, float a, float b) {
    // count up the length for the prior
        switch(o) {
            case PONE:   return 1+a;
            case INV:    return 1+a;
            case A:      return a; 
            case B:      return b; // need both since of how consts are passed in
            case PLUS:   return 1+a+b;
            case CPLUS:  return CONST_LENGTH_PRIOR+a;
            case MINUS:  return 1+a+b;
            case RMINUS: return 1+b+a;
            case TIMES:  return 1+a+b;
            case CTIMES: return CONST_LENGTH_PRIOR+a;
            case DIV:    return 1+a+b;
            case RDIV:   return 1+a+b;
            case SQRT:   return 1+a;
            case SQR:    return 1+a;
            case LOG:    return 1+a;
            case SIN:    return 1+a;
            case ASIN:   return 1+a;
            case ATAN:   return 1+a;
            case EXP:    return 1+a;
            case POW:    return 1+a+b;
            case CPOW:   return CONST_LENGTH_PRIOR+a;
            case RPOW:   return 1+b+a;
            case CRPOW:  return CONST_LENGTH_PRIOR+a;
            case GAMMA:  return 1+a;
            case BESSEL: return 1+a;
            case ABS:    return 1+a;
            default:     return CUDART_NAN_F;
        }    
}




__device__ float compute_prior(int N, int idx, op* P, float* C) {
    
    float len[PROGRAM_LENGTH+1]; // how long is the tree below? 
    
    for(int i=PROGRAM_LENGTH;i>=1;i--) { // start at the last node
        int lidx = 2*i; // indices of the children
        int ridx = 2*i+1;
        
        op o = P[idx+(i-1)*N];
        
        if(lidx>PROGRAM_LENGTH) len[i] = X_LENGTH_PRIOR; // x has length 1
        else                    len[i] = dispatch_length(o, len[lidx], len[ridx]);
        

    }
      
    float prior = -PRIOR_MULTIPLIER * len[1]; // the total length at the top

    for(int c=0;c<NCONSTANTS;c++) {
        prior += lcauchypdf(C[idx+c*N], CONSTANT_SCALE); // proportional to cauchy density
    }
    
    return prior;
    
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MCMC Kernel
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__device__ int is_descendant(int n, int k) {
    // this takes n,k as 0-indexed, but for the computation below we need them to be 1-indexed (I think?)
    
    if(k==n) return 1;
  
    // translate to 1-indexed
    n = n+1;
    k = k+1;
    
    // now we check if you're a descendant by whether or not a sequence of lambda x: 2x or lambda x:2x+1 will ever transform
    // k into n. This is the same as checking if k can be shifted over to be n
    while(k>n) {
        k = k >> 1; // shift once
        if(k == n) return 1;
    }
    
    return 0;
}

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
    
    op old_program[PROGRAM_LENGTH]; // store a buffer for the old program
    
	for(int mcmci=0;mcmci<steps;mcmci++) {
        
	    if( (mcmci & 0x1) == 0x1) { // propose to a structure every this often
                int n;//0-indexed 
                
                if(current > -CUDART_INF_F){
                    n = random_int(PROGRAM_LENGTH, RNG_ARGS); // pick a node to propose to, note that here 1 is the root (by tree logic)
                } else {
                    n = 0; // always propose to the root if we're terrible
                }
                                
                for(int i=n;i<PROGRAM_LENGTH;i++) {
                    
                    old_program[i] = P[idx+i*N]; // copy over
                    
//                     now make a proposal to every descendant of n
                    if(is_descendant(n,i)) { // translate i into 1-based indexing used for binary tree encoding.
                        P[i] = random_int(NOPS, RNG_ARGS);
                        // TODO: ADD If there is a constant below, propose from the prior?
                    }
                    
                }
                
                float proposal_prior = compute_prior(N, idx, P, C);
                float proposal_likelihood = compute_likelihood(N, idx, P, C, D, ndata);
                float proposal = proposal_prior + proposal_likelihood;
                        
                if((is_valid(proposal) && (proposal>current || random_float(RNG_ARGS) < expf(proposal-current))) || is_invalid(current)) {
                    current = proposal; // store the updated posterior
                    current_likelihood = proposal_likelihood;
                    current_prior = proposal_prior;
                } else {
                    
                    // restore 
                    for(int i=n;i<PROGRAM_LENGTH;i++) { // TODO: Could be started at n?
                        P[idx+i*N] = old_program[i]; 
                    }
                }

		
	    } 
        else { // propose to a constant otherwise
                
                int i = idx + N*random_int(NCONSTANTS, RNG_ARGS);
                float old = C[i];
                
                // choose what kind of proposal
                if(random_int(2,RNG_ARGS) == 0){
                    C[i] = C[i] + CONSTANT_SCALE*random_normal(RNG_ARGS); 
                } else {
                    C[i] = C[i] + CONSTANT_SCALE*random_cauchy(RNG_ARGS); 
                }
                // TODO: Maybe make proposals relative to the size of the constant?
                
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
//           
      } // end mcmc loop
	
    prior[idx] = current_prior; 
    likelihood[idx] = current_likelihood;
}




// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Output hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void string_dispatch( char* target, op o, const char* a, const char* b, const char* C) {
        switch(o) {
            case PONE:  strcat(target, "(1+"); strcat(target, a); strcat(target, ")"); break;
            case INV:   strcat(target, "(1/"); strcat(target, a); strcat(target, ")"); break;
            case A:     strcat(target, a); break;
            case B:     strcat(target, b); break;// need both since of how consts are passed in
            case PLUS:  strcat(target, "("); strcat(target, a); strcat(target, "+"); strcat(target, b); strcat(target, ")"); break;
            case CPLUS:  strcat(target, "("); strcat(target, a); strcat(target, "+"); strcat(target, C); strcat(target, ")"); break;
            case MINUS: strcat(target, "("); strcat(target, a); strcat(target, "-"); strcat(target, b); strcat(target, ")"); break;
            case RMINUS:strcat(target, "("); strcat(target, b); strcat(target, "-"); strcat(target, a); strcat(target, ")"); break;
            case TIMES: strcat(target, "("); strcat(target, b); strcat(target, "*"); strcat(target, a); strcat(target, ")"); break;
            case CTIMES: strcat(target, "("); strcat(target, b); strcat(target, "*"); strcat(target, C); strcat(target, ")"); break;
            case DIV:   strcat(target, "("); strcat(target, a); strcat(target, "/"); strcat(target, b); strcat(target, ")"); break;
            case RDIV:  strcat(target, "("); strcat(target, b); strcat(target, "/"); strcat(target, a); strcat(target, ")"); break;
            case LOG:   strcat(target, "log("); strcat(target, a); strcat(target, ")"); break;
            case SIN:   strcat(target, "sin("); strcat(target, a); strcat(target, ")"); break;
            case ASIN:  strcat(target, "asin("); strcat(target, a); strcat(target, ")"); break;
            case ATAN:  strcat(target, "atan("); strcat(target, a); strcat(target, ")"); break;
            case SQR:   strcat(target, "(("); strcat(target, a); strcat(target, ")^2)"); break;
            case SQRT:  strcat(target, "sqrt("); strcat(target, a); strcat(target, ")"); break;
            case EXP:   strcat(target, "exp("); strcat(target, a); strcat(target, ")"); break;
            case POW:   strcat(target, "("); strcat(target, a); strcat(target, "^"); strcat(target, b); strcat(target, ")"); break;
            case CPOW:  strcat(target, "("); strcat(target, a); strcat(target, "^"); strcat(target, C); strcat(target, ")"); break;
            case RPOW:  strcat(target, "("); strcat(target, b); strcat(target, "^"); strcat(target, a); strcat(target, ")"); break;
            case CRPOW: strcat(target, "("); strcat(target, C); strcat(target, "^"); strcat(target, a); strcat(target, ")"); break;
            case GAMMA: strcat(target, "gamma("); strcat(target,a); strcat(target, ")"); break;
            case BESSEL:strcat(target, "besselJ("); strcat(target, a); strcat(target, ",0)"); break;
            case ABS:   strcat(target, "abs("); strcat(target, a); strcat(target, ")"); break;
            default:    strcat(target, "NAN"); break;
        }    
}

char buf[2*PROGRAM_LENGTH+2][1000]; 
char cbuf[100];
void displaystring(const char* const_format, int N, int idx, op* P, float* C, FILE* fp){

    for(int i=PROGRAM_LENGTH;i>=1;i--) { // start at the last node
        int lidx = 2*i; // indices of the children
        int ridx = 2*i+1;

        strcpy(buf[i],""); // must zero since dispatch appends. 

        // put the xes there if we should
        if(lidx>PROGRAM_LENGTH) strcpy(buf[lidx],"x");
        if(ridx>PROGRAM_LENGTH) strcpy(buf[ridx],"x"); 
        
        sprintf(cbuf, const_format, C[idx+(i-1)*N]); // in case we need it
        string_dispatch(buf[i], P[idx+(i-1)*N], buf[lidx], buf[ridx], cbuf);
    }
    fprintf(fp, "%s", buf[1]);
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
    float resample_threshold = 100.0; // hypotheses more than this far from the MAP get resampled every outer block
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
    
    // -------------------------------------------------------------------------
    // Make the RNG replicable
    
    if(seed==-1) { srand(time(NULL)); seed = rand(); }

    // -------------------------------------------------------------------------
    // Log
    
    FILE* fp = fopen(SAMPLE_PATH.c_str(), "a");
    if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << SAMPLE_PATH <<"\n"; exit(1);}
    
    fprintf(fp, "# -----------------------------------------------------------------\n");
    fprintf(fp, "# -- Parameters:\n");
    fprintf(fp, "# -----------------------------------------------------------------\n");
    fprintf(fp, "# \tInput data path: %s\n", in_file_path.c_str());
    fprintf(fp, "# \tOutput path: %s\n", out_path.c_str());
    fprintf(fp, "# \tMCMC Iterations (per block): %i\n", steps);
    fprintf(fp, "# \tBlocks: %i\n", outer);
//     fprintf(fp, "\tBurn Blocks: %i\n", BURN_BLOCKS);
    fprintf(fp, "# \tN chains: %i\n", N);
    fprintf(fp, "# \tseed: %i\n", seed);
    fprintf(fp, "# \tMax program length: %i\n", PROGRAM_LENGTH);
    fprintf(fp, "# \tN Constants: %i\n", NCONSTANTS);
    fprintf(fp, "#\n#\n");
    
    // -----------------------------------------------------------------------
    // Read the data and set up some arrays
    // -----------------------------------------------------------------------
    
    vector<datum>* data_vec = load_data_file(in_file_path.c_str(), FIRST_HALF_DATA, EVEN_HALF_DATA);
    datum* host_D = &((*data_vec)[0]); // can do this with vectors now
    int ndata = data_vec->size();
    
    
    // Echo the data we actually run with (post-filtering for even/firsthalf)
    fprintf(fp, "# -----------------------------------------------------------------\n");
    fprintf(fp, "# -- Data:\n");
    fprintf(fp, "# -----------------------------------------------------------------\n");
    for(int i=0;i<ndata;i++) 
        fprintf(fp, "# %f\t%f\t%f\n", host_D[i].x, host_D[i].y, host_D[i].sd);
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
	    
            fprintf(fp, "%.4f\t%.4f\t%.4f\t", host_prior[h]+host_likelihood[h], host_prior[h], host_likelihood[h]);
//              
// //             for(int c=0;c<PROGRAM_LENGTH;c++){ 
// //                 fprintf(fp, "%d ", host_P[h+N*c]);
// //             }
// //             fprintf(fp, "\t\"");
// 
            fprintf(fp, "\"");
            displaystring("C", N, h, host_P, host_C, fp);
            
            fprintf(fp, "\"\t\"");
            
            displaystring("%.4f", N, h, host_P, host_C, fp);
            fprintf(fp, "\"\t\"");
            
            for(int i=0;i<PROGRAM_LENGTH;i++) {
                fprintf(fp, "%c", PROGRAM_CODE[host_P[h+N*i]]);
            }
            fprintf(fp, "\"\t\t");
            
            for(int c=0;c<NCONSTANTS;c++){ 
                fprintf(fp, "%.2f\t", host_C[h+N*c]);
            }
            
            fprintf(fp, "\n");
//             
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
        
        // print a progress bar to stderr
        int BAR_WIDTH = 70;
        fprintf(stderr, "\r[");
        for(int p=0;p<BAR_WIDTH;p++) { 
            if(p <= o * BAR_WIDTH/outer) fprintf(stderr,"=");
            else                         fprintf(stderr," ");
            
        }
        fprintf(stderr, "]");
    }
        
        
    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    fprintf(stderr, " Completed. \n");
    
    delete[] host_P;
    delete[] host_C;
    delete[] host_prior;
    delete[] host_likelihood;
    delete[] host_D;
    
}
