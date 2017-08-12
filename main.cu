/*
 * GPUropolis - 2017 June 20 - Steve Piantadosi 
 *
 * Simple tree-regeneration on CUDA with coalesced memory access
 *     
 * rename proposal, current
 * change power of 10 to power of 2, for speed
 * 
 */
#include <assert.h>
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

const float PRIOR_MULTIPLIER = 1.0; // prior is exp(-PRIOR_MULTIPLIER * length) 
const float CONST_LENGTH_PRIOR = 1.0; // how much do constants cost in terms of length?
const float PHYSICAL_CONSTANT_LENGTH_PRIOR = 1.0;
const float X_LENGTH_PRIOR = 1.0; // how much does X cost in terms of length?

const int PROGRAM_LENGTH = 31; // how many operations do we allow? This is the max index of the program array. 
const int NCONSTANTS     = 31; // each op may use a constant if it wants (but most do not)

const float CONSTANT_SCALE = 1.0; // Scales the constants in the prior
const int C_MAX_ORDER = 4; //  constants are chosen with scales between 10^C_MIN_ORDER and 10^C_MAX_ORDER
const int C_MIN_ORDER = -4; 
const float PROPOSE_INT = 0.1; // what proportion of the time do we propose to an integer?
const float P_PROPOSE_C_GEOMETRIC = 1.0; // how often do we propose setting a constant to an integer?
const float PROPOSE_GEOM_R = 0.1; // higher means we propose to higher integers (geometric rate on proposal)

const float RESAMPLE_QUANTILE = 0.5; // sort by posterior and resample everything less than this quantile

// Setup for gpu hardware 
const unsigned int BLOCK_SIZE = 128; // empirically determined to work well
int N_BLOCKS = 0; // set below
const unsigned int HARDARE_MAX_X_BLOCKS = 4096;
const unsigned int HARDWARE_MAX_THREADS_PER_BLOCK = 1024; // cannot exceed this many threads per block! For compute level 2.x and greater!

/* Define a few types so that we can change precision if we need to */
typedef int op; // what type is the program primitives?
typedef float data_t; // the type of the data we read and operate on in the program
typedef double bayes_t; // when we compute Bayesian things (priors, likelihoods, etc.) what type do we use?
typedef struct datum { data_t x; data_t y; data_t sd; } datum; // A structure to hold data: x,y,stdev pairs 

// Choose the set of operations
#if !defined SIMPLIFIED_OPS and !defined POLYNOMIAL_OPS and !defined PHYSICAL_CONSTANTS
//            C     I   a  b   +      -       _       #      *      @      /    |     L    E    ^     p    V      P     R     S     s    C    c      T    t     G      A    B
enum OPS {ONE, CONST, INV, A, B, PLUS, MINUS, RMINUS, CPLUS, TIMES, CTIMES, DIV, RDIV, LOG, EXP, POW, CPOW, RPOW, CRPOW, SQRT, SIN, ASIN, COS, ACOS, TAN, ATAN,  ABS, NOPS};
enum UNUSED_OPS { SQR=-999,  LOGOF2, HALF, E, PI, CLIGHT, G, HBAR, MU0, EL,  MP, ME, E0, KB, GAMMA, BESSEL};
const char* PROGRAM_CODE = "1CIab+-_#*@/|LE^pVPRSsCcTtA"; // if we want to print a concise description of the program (mainly for debugging) These MUST be algined with OPS
#endif

#if defined SIMPLIFIED_OPS
enum OPS {ONE, CONST, INV, A, B, PLUS, MINUS, RMINUS, CPLUS, TIMES, CTIMES, DIV, RDIV, NOPS};
enum UNUSED_OPS { SQR=-999,  LOGOF2, HALF, E, PI, CLIGHT, G, HBAR, MU0, EL,  MP, ME, E0, KB, ONE, LOG, EXP, POW, CPOW, RPOW, CRPOW, SQRT, SIN, ASIN, COS, ACOS, TAN, ATAN, GAMMA,  ABS, BESSEL};
const char* PROGRAM_CODE = "1CIab+-_#*@/|";
#endif

#if defined POLYNOMIAL_OPS
enum OPS {ONE, CONST, A, B, PLUS, MINUS, RMINUS, CPLUS, TIMES, CTIMES, CPOW, NOPS};
enum UNUSED_OPS { SQR=-999,  LOGOF2, HALF E, PI, CLIGHT, G, HBAR, MU0, EL,  MP, ME, E0, KB, ONE, LOG, EXP, POW,  RPOW, CRPOW, SQRT, SIN, ASIN, COS, ACOS, TAN, ATAN, GAMMA,  ABS, BESSEL, DIV, RDIV, INV};
const char* PROGRAM_CODE = "1CAB+-_#*@^";
#endif


#if defined LINEAR_OPS
enum OPS {ONE, CONST, A, B, PLUS, MINUS, RMINUS, CPLUS, CTIMES, NOPS};
enum UNUSED_OPS { SQR=-999, LOGOF2, HALF, E, PI, CLIGHT, G, HBAR, MU0, EL,  MP, ME, E0, KB, ONE, LOG, EXP, POW,  RPOW, CRPOW, SQRT, SIN, ASIN, COS, ACOS, TAN, ATAN, GAMMA,  ABS, BESSEL, DIV, RDIV, INV, TIMES, CPOW};
const char* PROGRAM_CODE = "1CAB+-_#@";
#endif


#if defined PHYSICAL_CONSTANTS
// see https://en.wikipedia.org/wiki/Physical_constant
// enum OPS {ONE, E, PI, CLIGHT, G, HBAR, MU0, EL,  MP, ME,   CONST, INV, A, B, PLUS, MINUS, RMINUS, CPLUS, TIMES, CTIMES, DIV, RDIV, LOG, EXP, POW, CPOW, RPOW, CRPOW, SQRT, SIN, ASIN, COS, ACOS, TAN, ATAN,  ABS,     NOPS };
// enum UNUSED_OPS { SQR=-999, GAMMA, BESSEL};
enum OPS {LOGOF2, HALF, E, PI, CLIGHT, G, HBAR, MU0, EL,  MP, ME, E0, KB, A,B,  PLUS, MINUS, RMINUS, TIMES, NOPS };

enum UNUSED_OPS { SQR=-999, ONE, GAMMA, BESSEL,  CONST, INV, LOG, EXP, POW, CPLUS,CTIMES, CPOW, RPOW, CRPOW, SQRT, SIN, ASIN, COS, ACOS, TAN, ATAN,  ABS,  DIV, RDIV };
const char* PROGRAM_CODE = "B2epcghm:o.~Kab+-_*";
#endif


// debugging
// enum OPS { CONST, A, B, CPLUS, CTIMES,   NOPS};
// enum UNUSED_OPS { BESSEL=-999, ONE, INV, PLUS, MINUS, RMINUS, TIMES, DIV, RDIV, LOG, EXP, POW, CPOW, RPOW, CRPOW, SQRT, SIN, ASIN, ATAN, GAMMA,  ABS, SQR,};
// const char* PROGRAM_CODE = "Cab+*"; // if we want to print a concise description of the program (mainly for debugging) These MUST be algined with OPS


// Command line arguments that get set below (these give default values)
data_t sdscale = 1.0; // scale the SDs by this
int N = 1024;
int steps = 1000;
int outer = 100;
int thin = 1; // how many outer blocks to skip?
int seed = -1;
int burn = 0;
int QUIET = 0; 
int WHICH_GPU = 0;
int FIRST_HALF_DATA = 0;
int EVEN_HALF_DATA = 0;
int SHOW_CONSTANTS = 0;
string in_file_path = "data.txt";
string out_path = "out/";
    
static struct option long_options[] =
    {   
        {"in",           required_argument,    NULL, 'd'},
        {"steps",   required_argument,    NULL, 'i'},
        {"N",            required_argument,    NULL, 'N'},
        {"out",          required_argument,    NULL, 'O'},
        {"outer",        required_argument,    NULL, 'o'},
        {"thin",         required_argument,    NULL, 't'},
        {"seed",         required_argument,    NULL, 's'},
        {"burn",         required_argument,    NULL, 'b'},
        {"gpu",          required_argument,    NULL, 'g'},
        {"sdscale",      required_argument,    NULL, 'r'},
        {"first-half",   no_argument,    NULL, 'f'},
        {"even-half",    no_argument,    NULL, 'e'},        
        {"show-constants",    no_argument,    NULL, 'C'},        
        {"quiet",        no_argument,    NULL, 'q'},
        {"all",    no_argument,    NULL, '_'},
        {NULL, 0, 0, 0} // zero row for bad arguments
    };  
    
int process_arguments(int argc, char** argv) {
        
    int option_index = 0, opt=0;
    while( (opt = getopt_long( argc, argv, "bp", long_options, &option_index )) != -1 )
        switch( opt ) {
            case 'd': in_file_path = optarg; break;
            case 'i': steps = atoi(optarg); break;
            case 'N': N = atoi(optarg); break;
            case 'o': outer = atoi(optarg); break;
            case 'O': out_path = optarg; break;
            case 't': thin = atoi(optarg); break;
            case 'b': burn = atoi(optarg); break;
            case 'g': WHICH_GPU = atoi(optarg); break;
            case 's': seed = atoi(optarg); break;
            case 'r': sdscale = (float)atof(optarg); break;
            case 'f': FIRST_HALF_DATA = 1; break;
            case 'e': EVEN_HALF_DATA = 1; break;
            case 'C': SHOW_CONSTANTS = 1; break;
            case 'q': QUIET = 1; break;
            case '_': break; // don't do anything if we use all the data
            
            default: 
                return 1; // unspecified
        }
    return 0;
}

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

#define is_valid(x) (!is_invalid(x))
#define is_invalid(x) (isnan(x) || (!isfinite(x))) 
// #define is_invalid(x) ( x==1.0/0.0 || x==0.0/0.0)
    

 #define max(a,b) \
   ({ typeof (a) _a = (a); \
      typeof (b) _b = (b); \
      _a > _b ? _a : _b; })
#define logplusexp(a,b) ({ typeof(a) m = max(a,b); logf(expf(a-m)+expf(b-m))+m; })
    
#define is_int(x) (abs(x-lroundf(x)) < 1.0e-6)
    
#define PIf 3.141592653589 
 
//
// Device random functions
// all have RNG_DEF (unlike host, which uses rand())
//
   
/* These macros allow us to use RNG_DEF and RNG_ARGS in function calls, and swap out 
 * for other variable sif we want to. e.g. 
 * 	int rand(int n, RNG_DEF){
 * 		....
 *	}
 * allows rx to be used inside. */
#define RNG_DEF int& rx
#define RNG_ARGS rx

#define MY_RAND_MAX ((1U << 31) - 1)


__device__ __host__ int cuda_rand(RNG_DEF) {
   //http://rosettacode.org/wiki/Linear_congruential_generator#C
   return rx = (rx * 1103515245 + 12345) & MY_RAND_MAX; 
}

__device__ int random_int(int n, RNG_DEF) {
     // number in [0,(n-1)]
    int divisor = MY_RAND_MAX/(n+1);
    int retval;

    do { 
        retval = cuda_rand(RNG_ARGS) / divisor;
    } while (retval >= n);

    return retval;
}

__device__ float random_float(RNG_DEF) {
    return float(random_int(1000000, RNG_ARGS)) / 1000000.0;
}


// Generate a 0-mean normal deviate with mean 0, sd 1
// This uses Box-muller so we don't need branching
__device__ float random_normal(RNG_DEF) {
    float u = random_float(RNG_ARGS);
    float v = random_float(RNG_ARGS);
    return sqrtf(-2.0*logf(u)) * sinf(2.0*PIf*v);
}

__device__ float random_cauchy(RNG_DEF) {
//     float u = random_float(RNG_ARGS);
//     return tanf(PIf*(u-0.5));
    float u = random_normal(RNG_ARGS);
    float v = random_normal(RNG_ARGS);
    return u/v;       
}
   
__device__ float random_geometric(float r, RNG_DEF) {
    // From Devroye Example 2.2
    return ceilf(logf(random_float(RNG_ARGS)) / log(r));
}


//
// Host random functions
//


__host__ int random_int(int limit) {
//  from http://stackoverflow.com/questions/2999075/generate-a-random-number-within-range/2999130#2999130
//   return a random number [0,limit).
 
    int divisor = RAND_MAX/limit;
    int retval;

    do { 
        retval = rand() / divisor;
    } while (retval > limit-1);

    return retval;
}

__host__ float random_float() {
    return float(rand())/RAND_MAX;
}

__host__ float random_normal() {
    float u = random_float();
    float v = random_float();
    return sqrtf(-2.0*log(u)) * sin(2.0*PIf*v);
}

__host__ float random_cauchy() {
    // in simulations in R, the CDF method does not work well compared to ratio of normals 
//     float u = random_float();
//     return tanf(PIf*(u-0.5));
    float u = random_normal();
    float v = random_normal();
    return u/v; 
}

//
// Probability densities
//
   
__device__ __host__ bayes_t lnormalpdf( float x, float s ){
    return -(x*x)/(2.0*s*s) - 0.5 * (logf(2.0*PIf) + 2.0*logf(s));
}

__device__ __host__ bayes_t lcauchypdf( float x, float s ){
    return -logf(PIf) - logf(s) - logf(1.0+x*x/s);
}

__device__ __host__ bayes_t lgeometricpdf(float n, float r) {
    return (n-1)*logf(r)+log(1.0-r);
}


// -----------------------------------------------------------------------
// Loading data
// -----------------------------------------------------------------------     
    
   
// Load data froma  file, putting it into our structs.
// This allows us to trim our data if we want
vector<datum>* load_data_file(const char* datapath, const int FIRST_HALF_DATA, const int EVEN_HALF_DATA) {

	FILE* fp = fopen(datapath, "r");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << datapath <<"\n"; exit(1);}
	
	vector<datum>* d = new vector<datum>();
	char* line = NULL; size_t len=0; 
	double x,y,sd; // load as doubles so scanf works and then cast to bayes_t 
	while( getline(&line, &len, fp) != -1) {
		if( line[0] == '#' ) continue;  // skip comments
		else if (sscanf(line, "%lf\t%lf\t%lf\n", &x, &y, &sd) == 3) { // floats
                        sd = sdscale*sd;
			d->push_back( (datum){.x=(data_t)x, .y=(data_t)y, .sd=(data_t)sd} );
                        assert((data_t)sd > 0.0);
		}
		else if (sscanf(line, "%le\t%le\t%le\n", &x, &y, &sd) == 3) { // scientific notation
                        sd = sdscale*sd;
			d->push_back( (datum){.x=(data_t)x, .y=(data_t)y, .sd=(data_t)sd} );
                        assert((data_t)sd > 0.0);
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
 

// define a template to fold any function over  
template<class T> __device__ __host__ T program_fold(int N, int idx, op* P, data_t* C, T leaf, T dispatch(op,T,T,T) ) {
   T buf[PROGRAM_LENGTH+1]; // we'll waste one space here at the beginning to simplify everything else
    
    for(int i=PROGRAM_LENGTH;i>=1;i--) { // start at the last node, but with +1 indexing
        int lidx = 2*i; // indices of the children, assuming 1-indexing
        int ridx = 2*i+1;
        
        T lvalue, rvalue;        
        if(lidx > PROGRAM_LENGTH) {
            lvalue = leaf; // the default value at the base of the tree
            rvalue = leaf;
        } else {
            lvalue = buf[lidx];
            rvalue = buf[ridx];
        }
        
        op o = P[idx+(i-1)*N];
	
        buf[i] = dispatch(o, lvalue, rvalue, C[idx+(i-1)*N]); // P[...-1] since P is zero-indexed
    }
    
    return buf[1];
}

// TODO: Make inline
// evaluat a single operation op on arguments a and b
__device__ data_t dispatch_eval(op o, data_t a, data_t b, data_t C) {
        switch(o) {
            case ONE:    return  1.0;
            case CONST:  return C;
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
            case SIN:    return sinf(a);
            case ASIN:   return asinf(a);
            case COS:    return cosf(a);
            case ACOS:   return acosf(a);
            case TAN:    return tanf(a);            
            case ATAN:   return atanf(a);
            case EXP:    return expf(a);
            case POW:    return powf(a,b);
            case CPOW:   return powf(a,C);
            case RPOW:   return powf(b,a);
            case CRPOW:  return powf(C,a);
            case GAMMA:  return tgammaf(a);
            case BESSEL: return j0f(a);
            case ABS:    return fabsf(a);
            case E:      return 1.6021766208e-19; // 1.6021766208e-19; // elementary charge
            case PI:     return 1.14472988584915; // 3.141592653589; // TAU/2
            case CLIGHT: return 19.5186009865452; // 299792458; // speed of light
            case G:      return -23.4302046558627; // 6.67408e-11; // gravitational constant
            case HBAR:   return -78.2347583540371; // 1.054571800e-34; // reduced planck's constant
            case MU0:    return -13.5870714043359; // 1.256637061e-6; // magnetic constant
            case EL:     return -43.2777536741305; // 1.6021766208e-19; //elementary charge
            case MP:     return -61.6554051167679; // 1.672621898e-27; // proton mass
            case ME:     return -69.1708328401338; // 9.10938356e-31; // electron mass
            case E0:     return -25.4501305691713; // epsilon_0, vacuum permittivity
            case KB:     return -52.6369038080531; // bolzman constant
            case LOGOF2: return log(2.0);
            case HALF:   return a/2.0;
            default:     return CUDART_NAN_F;
        }    
}

__device__ data_t call(int N, int idx, op* P, data_t* C, data_t x) {
  return program_fold<data_t>(N, idx, P, C, x, dispatch_eval);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the likelihood
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ bayes_t compute_likelihood(int N, int idx, op* P, data_t* C, datum* D, int ndata) {
    bayes_t ll = 0.0; // total ll 
    for(int di=0;di<ndata;di++) {
        
        data_t fx = call(N, idx, P, C, D[di].x);
        if(is_invalid(fx)) return -CUDART_INF_F;
	  
        bayes_t l = lnormalpdf(fx-D[di].y, D[di].sd);
        if(is_invalid(l)) return -CUDART_INF_F;
        
        ll += l;
    }
    return ll;
    
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the prior
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__device__ __host__ float dispatch_length(op o, float a, float b, float c__unusued) {
    // count up the length for the prior
    // c is unused
        switch(o) {
            case ONE:    return 1;
            case CONST:  return CONST_LENGTH_PRIOR;
            case INV:    return 1+a;
            case A:      return a; 
            case B:      return b; // need both since of how consts are passed in
            case PLUS:   return 1+a+b;
            case CPLUS:  return CONST_LENGTH_PRIOR+a+1;
            case MINUS:  return 1+a+b;
            case RMINUS: return 1+b+a;
            case TIMES:  return 1+a+b;
            case CTIMES: return CONST_LENGTH_PRIOR+a+1;
            case DIV:    return 1+a+b;
            case RDIV:   return 1+a+b;
            case SQRT:   return 1+a;
            case SQR:    return 1+a;
            case LOG:    return 1+a;
            case SIN:    return 1+a;
            case ASIN:   return 1+a;
            case COS:    return 1+a;
            case ACOS:   return 1+a;
            case TAN:    return 1+a;
            case ATAN:   return 1+a;
            case EXP:    return 1+a;
            case POW:    return 1+a+b;
            case CPOW:   return CONST_LENGTH_PRIOR+a+1;
            case RPOW:   return 1+b+a;
            case CRPOW:  return CONST_LENGTH_PRIOR+a+1;
            case GAMMA:  return 1+a;
            case BESSEL: return 1+a;
            case ABS:    return 1+a;
            case E:      return PHYSICAL_CONSTANT_LENGTH_PRIOR; // elementary charge
            case PI:     return PHYSICAL_CONSTANT_LENGTH_PRIOR; // TAU/2
            case CLIGHT: return PHYSICAL_CONSTANT_LENGTH_PRIOR; // speed of light
            case G:      return PHYSICAL_CONSTANT_LENGTH_PRIOR; // gravitational constant
            case HBAR:   return PHYSICAL_CONSTANT_LENGTH_PRIOR; // reduced planck's constant
            case MU0:    return PHYSICAL_CONSTANT_LENGTH_PRIOR; // magnetic constant
            case EL:     return PHYSICAL_CONSTANT_LENGTH_PRIOR; //elementary charge
            case MP:     return PHYSICAL_CONSTANT_LENGTH_PRIOR; // proton mass
            case ME:     return PHYSICAL_CONSTANT_LENGTH_PRIOR; // electron mass
            case LOGOF2: return PHYSICAL_CONSTANT_LENGTH_PRIOR;
            case HALF:   return 1+a;
            case E0:     return PHYSICAL_CONSTANT_LENGTH_PRIOR;
            case KB:     return PHYSICAL_CONSTANT_LENGTH_PRIOR;
            default:     return 0.0/0.0; // nan 
        }    
}


__device__ bayes_t compute_prior(int N, int idx, op* P, data_t* C) {
   
    // call fold where the X_LENGTH_PRIOR is used for the leaves
    float length = program_fold<float>(N, idx, P, C, X_LENGTH_PRIOR, dispatch_length);
 
    // the prior is going to be (1/NOPS)**(PRIOR_MULTIPLIER * LENGTH)
    
    bayes_t prior = -PRIOR_MULTIPLIER * length * NOPS;
    for(int c=0;c<NCONSTANTS;c++) {
        prior += lcauchypdf(C[idx+c*N], CONSTANT_SCALE); // proportional to cauchy density
//         prior += lnormalpdf(C[idx+c*N], CONSTANT_SCALE);
    }
    
    return prior;
    
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// If we want to compute polynomial degrees
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ float dispatch_degree(op o, float a, float b, float c) {
    // give the polynomial degree if it only uses polynomial ops,
    // otherwise INFINITY
    
    switch(o) {
            case ONE:    return 0;
            case CONST:  return 0;
            case A:      return a; 
            case B:      return b;
            case PLUS:   return max(a,b);
            case CPLUS:  return a;
            case MINUS:  return max(a,b);
            case RMINUS: return max(a,b);
            case TIMES:  return a+b;
            case CTIMES: return a;
            case CPOW:   return a*c; 
            case POW:    return (b==0.0) ? a*c : 1.0/0.0; // x^C
            case DIV:    return (b==0.0) ? a : 1.0/0.0; // we can divide by constants
            case E:      return 0; // elementary charge
            case PI:     return 0; // TAU/2
            case CLIGHT: return 0; // speed of light
            case G:      return 0; // gravitational constant
            case HBAR:   return 0; // reduced planck's constant
            case MU0:    return 0; // magnetic constant
            case EL:     return 0; //elementary charge
            case MP:     return 0; // proton mass
            case ME:     return 0; // electron mass  
            case LOGOF2: return 0; 
            case HALF:   return a;
            case E0:     return 0;
            case KB:     return 0;
            
            // TODO: WE CAN INCLDUE THINGS LIKE COS(C), we just have to pass zeros through, not xes
            
            default:     return 1.0/0.0; // return inf 
        }    
}
__host__ float polynomial_degree(int N, int idx, op* P, data_t* C) {
  return program_fold<float>(N, idx, P, C, 1, dispatch_degree);
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

__global__ void MH_simple_kernel(int N, op* P, data_t* C, datum* D, int ndata, int steps, bayes_t* prior, bayes_t* likelihood, int random_seed){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) { return; }  // MUST have this or else all hell breaks loose

    bayes_t current_prior = compute_prior(N, idx, P, C);
    bayes_t current_likelihood = compute_likelihood(N, idx, P, C, D, ndata);
    bayes_t current_posterior = current_prior + current_likelihood;
        
    // Two possibilities here. We could let everything do the same proposals (for all steps) in which case we don't
    // add idx. This makes them access the same memory, etc. Alternatively we could add idx and make them separate
    // doesn't seem to matter much for speed
    // int rx = random_seed;     	
    int rx = random_seed + idx; // NOTE: We might want to call cuda_rand here once so that we don't have a bias from low seeds
    
    op old_program[PROGRAM_LENGTH]; // store a buffer for the old program
    data_t old_C[NCONSTANTS];       // and old constants
    
	for(int mcmci=0;mcmci<steps;mcmci++) {
        
	    if( mcmci % 5 == 0) { // propose to a structure every this often
                int n;//0-indexed 
                
                if(is_invalid(current_posterior)){
                   n = 0; // always propose to the root if we're terrible
                } else {
                   n = random_int(PROGRAM_LENGTH, RNG_ARGS); // pick a node to propose to, note that here 1 is the root (by tree logic)                 
                }
                                                
                for(int i=n;i<PROGRAM_LENGTH;i++) {
                    
                    old_program[i] = P[idx+i*N]; // copy over
                    
                    // now make a proposal to every descendant of n
                    if(is_descendant(n,i)) { // translate i into 1-based indexing used for binary tree encoding.
                        P[idx+i*N] = random_int(NOPS, RNG_ARGS);
                    }
                    
                }
                
                bayes_t proposal_prior = compute_prior(N, idx, P, C);
                bayes_t proposal_likelihood = compute_likelihood(N, idx, P, C, D, ndata);
                bayes_t proposal_posterior = proposal_prior + proposal_likelihood;
                
                // TODO: Check here -- do we have to count the tree size?
                        
                if(is_valid(proposal_posterior) && (proposal_posterior>current_posterior || random_float(RNG_ARGS) < expf(proposal_posterior-current_posterior))) {
                    current_posterior = proposal_posterior; // store the updated posterior
                    current_likelihood = proposal_likelihood;
                    current_prior = proposal_prior;
                } else {
                    
                    //restore 
                    for(int i=n;i<PROGRAM_LENGTH;i++) { // TODO: Could be started at n?
                        P[idx+i*N] = old_program[i]; 
                    }
                }
		
        } 
        else { // propose to a constant otherwise
            bayes_t forward = 0.0, backward = 0.0; //log forward - log backward probability
            
            // In this version, we propose to all constants, but with varying scales
            for(int c=0;c<NCONSTANTS;c++) {
                int i = idx+N*c;  
                old_C[c] = C[i];
                
                // sometimes we propose changing to zero, or away from zero to the prior
                if(random_float(RNG_ARGS) < P_PROPOSE_C_GEOMETRIC) {
                    
                    /* To deal with the unused constants, here we mix tegether a random_geometric and a normal
                     * This is necessary or else the prior of the unused constants matters a lot. 
                     */
                    if(random_float(RNG_ARGS) < PROPOSE_INT) { // propose to an integer
		        float sgn = random_float(RNG_ARGS) < 0.5 ? -1.0 : 1.0; // -1 or 1
		        
                        C[i] = sgn*random_geometric(PROPOSE_GEOM_R, RNG_ARGS)-1; // shift by 1 to make 0 the least of support
                        
                        forward += lgeometricpdf(C[i]+1, PROPOSE_GEOM_R) + (C[i]==0.0 ? 0.0 : log(2.0)) + log(PROPOSE_INT); // forward, counting two ways to get to 0.0 (+/-)
                    } else { // propose from the prior 
                        C[i] = CONSTANT_SCALE * random_normal(RNG_ARGS);  
                        forward += lnormalpdf(C[i]/CONSTANT_SCALE, 1.0) + log(1.0-PROPOSE_INT); // forward
                    }

                    // the backward
                    if( is_int(old_C[c]) ) { 
                        // two ways to go back if it's an int--a proposal from the prior or an integer proposal
                        backward += logplusexp(lgeometricpdf(old_C[c]+1, PROPOSE_GEOM_R) + (C[i]==0.0 ? 0.0 : log(2.0)) + log(PROPOSE_INT), 
                                               lnormalpdf(old_C[c]/CONSTANT_SCALE, 1.0) + log(1.0-PROPOSE_INT));
                    } else {
                        // otherwise we can only go back with a prior proposal
                        backward += lnormalpdf(old_C[c]/CONSTANT_SCALE, 1.0) + log(1.0-PROPOSE_INT);
                    }
                    
                    
                }
                else{
                    // Most of our proposals are just normal centered on the current value.                     
                    int order = random_int(C_MAX_ORDER-C_MIN_ORDER,RNG_ARGS)+C_MIN_ORDER; // what order of magnitude?
                    C[i] = C[i] + CONSTANT_SCALE * powf(10.0,order) * random_normal(RNG_ARGS);        
                    // proposal is symmetric so no forward/backward
                }
            }
            
            bayes_t proposal_prior = compute_prior(N, idx, P, C);
            bayes_t proposal_likelihood = compute_likelihood(N, idx, P, C, D, ndata);
            bayes_t proposal_posterior = proposal_prior + proposal_likelihood;
            
            bayes_t acceptance = proposal_posterior + backward - forward;
            
            if((is_valid(acceptance) && (proposal_posterior>current_posterior || random_float(RNG_ARGS) < expf(acceptance))) || is_invalid(current_posterior)) {
                current_posterior = proposal_posterior; // store the updated posterior
                current_likelihood = proposal_likelihood;
                current_prior = proposal_prior;
            } else {
                /// on reject we restore
                for(int c=0;c<NCONSTANTS;c++) {
                    int i = idx+N*c;  
                    C[i] = old_C[c];
                }
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
            case ONE:   strcat(target, "1"); break;
            case CONST: strcat(target, C); break;
            case INV:   strcat(target, "(1/"); strcat(target, a); strcat(target, ")"); break;
            case A:     strcat(target, a); break;
            case B:     strcat(target, b); break;// need both since of how consts are passed in
            case PLUS:  strcat(target, "("); strcat(target, a); strcat(target, "+"); strcat(target, b); strcat(target, ")"); break;
            case CPLUS: strcat(target, "("); strcat(target, a); strcat(target, "+"); strcat(target, C); strcat(target, ")"); break;
            case MINUS: strcat(target, "("); strcat(target, a); strcat(target, "-"); strcat(target, b); strcat(target, ")"); break;
            case RMINUS:strcat(target, "("); strcat(target, b); strcat(target, "-"); strcat(target, a); strcat(target, ")"); break;
            case TIMES: strcat(target, "("); strcat(target, b); strcat(target, "*"); strcat(target, a); strcat(target, ")"); break;
            case CTIMES:strcat(target, "("); strcat(target, a); strcat(target, "*"); strcat(target, C); strcat(target, ")"); break;
            case DIV:   strcat(target, "("); strcat(target, a); strcat(target, "/"); strcat(target, b); strcat(target, ")"); break;
            case RDIV:  strcat(target, "("); strcat(target, b); strcat(target, "/"); strcat(target, a); strcat(target, ")"); break;
            case LOG:   strcat(target, "log("); strcat(target, a); strcat(target, ")"); break;
            case SIN:   strcat(target, "sin("); strcat(target, a); strcat(target, ")"); break;
            case ASIN:  strcat(target, "asin("); strcat(target, a); strcat(target, ")"); break;
            case COS:   strcat(target, "cos("); strcat(target, a); strcat(target, ")"); break;
            case ACOS:  strcat(target, "acos("); strcat(target, a); strcat(target, ")"); break;
            case TAN:   strcat(target, "tan("); strcat(target, a); strcat(target, ")"); break;
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
            
            
            case E:      strcat(target, "E"); break; // elementary charge
            case PI:     strcat(target, "PI"); break; // TAU/2
            case CLIGHT: strcat(target, "CLIGHT"); break; // speed of light
            case G:      strcat(target, "G"); break; // gravitational constant
            case HBAR:   strcat(target, "HBAR"); break; // reduced planck's constant
            case MU0:    strcat(target, "MU0"); break; // magnetic constant
            case EL:     strcat(target, "EL"); break; //elementary charge
            case MP:     strcat(target, "MP"); break; // proton mass
            case ME:     strcat(target, "ME"); break; // electron mass
            case E0:     strcat(target, "E0"); break; // electric constant (vacuum permitivity)
            case KB:     strcat(target, "KB"); break; // bolzman constant
            
            case LOGOF2: strcat(target, "log(2.0)"); break;
            case HALF:   strcat(target, "("); strcat(target, a); strcat(target, "/2.0)"); break;
            
            
            default:    strcat(target, "NAN"); break;
        }    
}

char buf[2*PROGRAM_LENGTH+2][1000]; 
char lbuf[100], rbuf[100];
char cbuf[100];
void displaystring(const char* const_format, int N, int idx, op* P, data_t* C, FILE* fp){

    for(int i=PROGRAM_LENGTH;i>=1;i--) { // start at the last node
        int lidx = 2*i; // indices of the children
        int ridx = 2*i+1;

        strcpy(buf[i],""); // must zero since dispatch appends. 

        // put the xes there if we should
        if(lidx>PROGRAM_LENGTH) strcpy(lbuf,"x");
        else                    strcpy(lbuf, buf[lidx]);
        
        if(ridx>PROGRAM_LENGTH) strcpy(rbuf,"x"); 
        else                    strcpy(rbuf, buf[ridx]);
        
        sprintf(cbuf, const_format, C[idx+(i-1)*N]); // in case we need it
        
        string_dispatch(buf[i], P[idx+(i-1)*N], lbuf, rbuf, cbuf);
    }
    fprintf(fp, "%s", buf[1]);
}


void latex_dispatch( char* target, op o, const char* a, const char* b, const char* C) {
        switch(o) {
            case ONE:   strcat(target, "1"); break;
            case CONST: strcat(target, C); break;
            case INV:   strcat(target, "1/"); strcat(target, a); strcat(target, ""); break;
            case A:     strcat(target, a); break;
            case B:     strcat(target, b); break;// need both since of how consts are passed in
            case PLUS:  strcat(target, ""); strcat(target, a); strcat(target, "+"); strcat(target, b); strcat(target, ""); break;
            case CPLUS: strcat(target, ""); strcat(target, a); strcat(target, "+"); strcat(target, C); strcat(target, ""); break;
            case MINUS: strcat(target, ""); strcat(target, a); strcat(target, "-"); strcat(target, b); strcat(target, ""); break;
            case RMINUS:strcat(target, ""); strcat(target, b); strcat(target, "-"); strcat(target, a); strcat(target, ""); break;
            case TIMES: strcat(target, ""); strcat(target, b); strcat(target, " "); strcat(target, a); strcat(target, ""); break;
            case CTIMES:strcat(target, ""); strcat(target, a); strcat(target, " \\cdot "); strcat(target, C); strcat(target, ""); break;
            case DIV:   strcat(target, "\\frac{"); strcat(target, a); strcat(target, "}{"); strcat(target, b); strcat(target, "}"); break;
            case RDIV:  strcat(target, "\\frac{"); strcat(target, b); strcat(target, "}{"); strcat(target, a); strcat(target, "}"); break;
            case LOG:   strcat(target, "\\log("); strcat(target, a); strcat(target, ")"); break;
            case SIN:   strcat(target, "sin("); strcat(target, a); strcat(target, ")"); break;
            case ASIN:  strcat(target, "asin("); strcat(target, a); strcat(target, ")"); break;
            case COS:   strcat(target, "cos("); strcat(target, a); strcat(target, ")"); break;
            case ACOS:  strcat(target, "acos("); strcat(target, a); strcat(target, ")"); break;
            case TAN:   strcat(target, "tan("); strcat(target, a); strcat(target, ")"); break;
            case ATAN:  strcat(target, "atan("); strcat(target, a); strcat(target, ")"); break;
            case SQR:   strcat(target, "(("); strcat(target, a); strcat(target, ")^2)"); break;
            case SQRT:  strcat(target, "\\sqrt{"); strcat(target, a); strcat(target, "}"); break;
            case EXP:   strcat(target, "e^{"); strcat(target, a); strcat(target, "}"); break;
            case POW:   strcat(target, ""); strcat(target, a); strcat(target, "^{"); strcat(target, b); strcat(target, "}"); break;
            case CPOW:  strcat(target, ""); strcat(target, a); strcat(target, "^{"); strcat(target, C); strcat(target, "}"); break;
            case RPOW:  strcat(target, ""); strcat(target, b); strcat(target, "^{"); strcat(target, a); strcat(target, "}"); break;
            case CRPOW: strcat(target, ""); strcat(target, C); strcat(target, "^{"); strcat(target, a); strcat(target, "}"); break;
            case GAMMA: strcat(target, "\\Gamma("); strcat(target,a); strcat(target, ")"); break;
            case BESSEL:strcat(target, "J_0("); strcat(target, a); strcat(target, ",0)"); break;
            case ABS:   strcat(target, "\\mid "); strcat(target, a); strcat(target, "\\mid "); break;
            
            
            case E:      strcat(target, "e"); break; // elementary charge
            case PI:     strcat(target, "\\pi"); break; // TAU/2
            case CLIGHT: strcat(target, "c"); break; // speed of light
            case G:      strcat(target, "G"); break; // gravitational constant
            case HBAR:   strcat(target, "\\bar{h}"); break; // reduced planck's constant
            case MU0:    strcat(target, "\\mu_0"); break; // magnetic constant
            case EL:     strcat(target, "e_l"); break; //elementary charge
            case MP:     strcat(target, "m_p"); break; // proton mass
            case ME:     strcat(target, "m_e"); break; // electron mass
            case E0:     strcat(target, "\\epsilon_0"); break; // electron mass
            case KB:     strcat(target, "k_B"); break; // bolzman constant
            
            case LOGOF2: strcat(target, "\\log 2)"); break;
            case HALF:   strcat(target, "("); strcat(target, a); strcat(target, "/2.0)"); break;
            
            
            default:    strcat(target, "NAN"); break;
        }    
}


void display_latex(const char* const_format, int N, int idx, op* P, data_t* C, FILE* fp){

    for(int i=PROGRAM_LENGTH;i>=1;i--) { // start at the last node
        int lidx = 2*i; // indices of the children
        int ridx = 2*i+1;

        strcpy(buf[i],""); // must zero since dispatch appends. 

        // put the xes there if we should
        if(lidx>PROGRAM_LENGTH) strcpy(lbuf,"x");
        else                    strcpy(lbuf, buf[lidx]);
        
        if(ridx>PROGRAM_LENGTH) strcpy(rbuf,"x"); 
        else                    strcpy(rbuf, buf[ridx]);
        
        sprintf(cbuf, const_format, C[idx+(i-1)*N]); // in case we need it
        
        latex_dispatch(buf[i], P[idx+(i-1)*N], lbuf, rbuf, cbuf);
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

// for comparing bayes_t types (computing the median)
int cmp(const void *a, const void* b) { 
    bayes_t ap = *static_cast<const bayes_t*>(a);
    bayes_t bp = *static_cast<const bayes_t*>(b); 
    
    // use != here to check for nan:
    if( ap>bp || (bp!=bp) ) { return 1; }
    if( ap<bp || (ap!=ap) ) { return -1; }
    return 0;
}

const int PROGRESS_BAR_WIDTH = 70;
void progress_bar(float pct) {
    fprintf(stderr, "\r[");
    for(int p=0;p<PROGRESS_BAR_WIDTH;p++) { 
        if(p <= pct * PROGRESS_BAR_WIDTH) 
            fprintf(stderr,"=");
        else 
            fprintf(stderr," ");
        
    }
    fprintf(stderr, "]");
}


int main(int argc, char** argv)
{   
    // ----------------------------------------------------------------------------
    // Parse command line
    // -----------------------------------------------------------------------
   
    if(process_arguments(argc, argv)==1) {
        cerr << "Invalid command line arguments." << endl;
        return 1;
    }
        
    // -----------------------------------------------------------------------
    // Check a few things
    // -----------------------------------------------------------------------
    
    assert(strlen(PROGRAM_CODE) == NOPS);
    assert(PROGRAM_LENGTH == NCONSTANTS);
    assert(steps>0);
    assert(N>0);
    assert(burn>=0);
    assert(thin>=1);
    
    // -----------------------------------------------------------------------
    // Initialize the GPU
    // -----------------------------------------------------------------------
     
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if(WHICH_GPU <= deviceCount) {
        cudaError_t err = cudaSetDevice(WHICH_GPU);
        if(err != cudaSuccess)
            printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    else {
        cerr << "Invalid GPU device " << WHICH_GPU << endl;
        return 1;
    }
    int wgpu; cudaGetDevice(&wgpu);
    
    
    cudaDeviceReset();
    
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
    
    if(seed==-1) { 
        seed = time(NULL);
    }
   
    // -----------------------------------------------------------------------
    // Read the data and set up some arrays
    // -----------------------------------------------------------------------
    
    vector<datum>* data_vec = load_data_file(in_file_path.c_str(), FIRST_HALF_DATA, EVEN_HALF_DATA);
    datum* host_D = &((*data_vec)[0]); // can do this with vectors now
    int ndata = data_vec->size();
    
    
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
    fprintf(fp, "# \tSeed: %i\n", seed);
    fprintf(fp, "# \tMax program length: %i\n", PROGRAM_LENGTH);
    fprintf(fp, "# \tN Constants: %i\n", NCONSTANTS);
    fprintf(fp, "# \tGPU: %i\n", wgpu);
    fprintf(fp, "# \tNumber of data points: %i\n", ndata);
    fprintf(fp, "#\n#\n");
    
    // set up the seed
    srand(seed);
    
    
    // Echo the data we actually run with (post-filtering for even/firsthalf)
    fprintf(fp, "# -----------------------------------------------------------------\n");
    fprintf(fp, "# -- Data:\n");
    fprintf(fp, "# -----------------------------------------------------------------\n");
    for(int i=0;i<ndata;i++) 
        fprintf(fp, "# \t%f\t%f\t%f\n", host_D[i].x, host_D[i].y, host_D[i].sd);
    fprintf(fp, "#\n#\n");
    fclose(fp);
    
    // -----------------------------------------------------------------------
    // Set up all the programs locally
    // -----------------------------------------------------------------------
    
    op*    host_P;             cudaMallocHost((void**)&host_P, sizeof(op)*N*PROGRAM_LENGTH); // new op[N*PROGRAM_LENGTH];
    CUDA_CHECK();
    data_t* host_C;            cudaMallocHost((void**)&host_C, sizeof(data_t)*N*NCONSTANTS); // new data_t[N*NCONSTANTS];
    CUDA_CHECK();
    bayes_t* host_prior;       cudaMallocHost((void**)&host_prior, sizeof(bayes_t)*N); // new bayes_t[N];
    CUDA_CHECK();
    bayes_t* host_likelihood;  cudaMallocHost((void**)&host_likelihood, sizeof(bayes_t)*N); // new bayes_t[N]; 
    CUDA_CHECK();

    
    for(int i=0;i<PROGRAM_LENGTH*N;i++) host_P[i] = random_int(NOPS-1);
    for(int i=0;i<NCONSTANTS*N;i++)    {
        int order = random_int(C_MAX_ORDER-C_MIN_ORDER)+C_MIN_ORDER; 
        host_C[i] = CONSTANT_SCALE * powf(10.0,order) * random_normal();
    }
    
    // -----------------------------------------------------------------------
    // Allocate on device and copy
    // -----------------------------------------------------------------------

    DEVARRAY(datum, D, ndata) // defines device_D, 
    DEVARRAY(op,    P, N*PROGRAM_LENGTH) // device_P
    DEVARRAY(data_t, C, N*NCONSTANTS) // device_C
    DEVARRAY(bayes_t, prior, N) 
    DEVARRAY(bayes_t, likelihood, N) 
    
    // -----------------------------------------------------------------------
    // Run
    // -----------------------------------------------------------------------
    
    if(! QUIET) {progress_bar(0.0); }
    
    // main outer loop
    for(int o=0;o<outer;o++) {
        
        // run this many steps
        // interesting, without the rand() call, we have eventual decreases in posterior probability, 
        MH_simple_kernel<<<N_BLOCKS,BLOCK_SIZE>>>(N, device_P, device_C, device_D, ndata, steps, device_prior, device_likelihood, rand());
    
        // copy memory back 
        cudaMemcpy(host_P, device_P, N*sizeof(op)*PROGRAM_LENGTH, cudaMemcpyDeviceToHost); CUDA_CHECK();
        cudaMemcpy(host_C, device_C, N*sizeof(data_t)*NCONSTANTS, cudaMemcpyDeviceToHost); CUDA_CHECK();
        cudaMemcpy(host_prior, device_prior, N*sizeof(bayes_t), cudaMemcpyDeviceToHost); CUDA_CHECK();
        cudaMemcpy(host_likelihood, device_likelihood, N*sizeof(bayes_t), cudaMemcpyDeviceToHost); CUDA_CHECK();
        cudaDeviceSynchronize(); // wait for preceedings requests to finish
        
        // print every thin samples
        if(o>burn && o%thin == 0) {
            // and now print
            fp = fopen(SAMPLE_PATH.c_str(), "a");
            for(int h=0;h<N;h++) {
                fprintf(fp, "%d\t%d\t", h, o);
                
                fprintf(fp, "%.3f\t%.3f\t%.3f\t", host_prior[h]+host_likelihood[h], host_prior[h], host_likelihood[h]);
 
                fprintf(fp, "%.1f\t",  program_fold<float>(N, h, host_P, host_C, X_LENGTH_PRIOR, dispatch_length));
                
                // print the degree
                float deg = polynomial_degree(N, h, host_P, host_C);
                if(deg == 1.0/0.0) { fprintf(fp, "NA\t"); }
                else{                fprintf(fp, "%.2f\t", deg);}

                
                fprintf(fp, "\"");
                displaystring("C", N, h, host_P, host_C, fp);
                
                fprintf(fp, "\"\t\"");
                
                displaystring("%.10f", N, h, host_P, host_C, fp);
                fprintf(fp, "\"\t\"");
                
                display_latex("%.2f", N, h, host_P, host_C, fp);
                fprintf(fp, "\"\t\"");
                
                
                for(int i=0;i<PROGRAM_LENGTH;i++) {
                    fprintf(fp, "%c", PROGRAM_CODE[host_P[h+N*i]]);
                }
                fprintf(fp, "\"\t");    
                
                
                if(SHOW_CONSTANTS) {
                    for(int c=0;c<NCONSTANTS;c++){ 
                        fprintf(fp, "%.10f\t", host_C[h+N*c]);
                    }
                }
                
                fprintf(fp, "\n");            
            }
            fclose(fp);
        }
        
        // find the cutoff 
	bayes_t host_posterior[N];
	for(int h=0;h<N;h++) host_posterior[h] = host_prior[h]+host_likelihood[h];
	qsort( (void*)host_posterior, N, sizeof(bayes_t), cmp);
  	bayes_t cutoff = host_posterior[int(N*RESAMPLE_QUANTILE)];
        
        // Now resample from the prior if we are too bad -- worse than the cutoff
        for(int h=0;h<N;h++) {
            if(host_prior[h]+host_likelihood[h] < cutoff) {
                // randomize
                for(int i=0;i<PROGRAM_LENGTH;i++) host_P[h+i*N] = random_int(NOPS-1);
                for(int i=0;i<NCONSTANTS;i++)   {
                    int order = random_int(C_MAX_ORDER-C_MIN_ORDER)+C_MIN_ORDER; 
                    host_C[h+i*N] = CONSTANT_SCALE * powf(10.0,order) * random_normal();    
                }
            }
        }
        // and copy back 
        cudaMemcpy(device_P, host_P, N*PROGRAM_LENGTH*sizeof(op), cudaMemcpyHostToDevice);  CUDA_CHECK();
        cudaMemcpy(device_C, host_C, N*NCONSTANTS*sizeof(data_t), cudaMemcpyHostToDevice);  CUDA_CHECK();
        
        // print a progress bar to stderr
        if(! QUIET) {progress_bar(float(o)/float(outer)); }
          
    }
    
    
    // finish printing this since it makes steve happy
    if(! QUIET) {progress_bar(1.1); }
        
    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    
    fprintf(stderr, " Completed. \n");
    
    cudaFreeHost(host_P);
    cudaFreeHost(host_C);
    cudaFreeHost(host_prior);
    cudaFreeHost(host_likelihood);
    delete[] host_D;
    
}
