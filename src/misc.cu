/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Assorted misc functions
 */

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Some math
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// use a define so we can work on host and device
#define PIf 3.141592653589

// Define -log(x) or log(1/x)
#define nlog(x)  (-log(x))
#define nlogf(x) (-log(float(x)))
#define logf(x)  log(float(x))
#define mean(x,y) (((x)+(y))/2)

// Needed to define CUDART_NAN_F
#include "math_constants.h"

// check nans. isfinite requires finite and non-nan
#define is_valid(x) (isfinite(x))
#define is_invalid(x) (!is_valid(x))

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CUDA
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

void cudaSynchronizeAndErrorCheck() {
	cudaDeviceSynchronize(); //
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(error)); }
	
}	

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Random number generation 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// And some macros to help with random numbers, so we can concisely pass these around functions etc
// #define RNG_DEF int& rx, int& ry, int& rz, int& rw
// #define RNG_ARGS rx,ry,rz,rw
// 
#define RNG_DEF int& rx
#define RNG_ARGS rx
#define MY_RAND_MAX_32 ((1U << 31) - 1)
#define MY_RAND_MAX ((1U << 15) - 1)
// for use on the host:

// A linear congruential 
 __device__ __host__ int random_int(int n, RNG_DEF) {

	//http://rosettacode.org/wiki/Linear_congruential_generator#C
	int r = (rx = (rx * 214013 + 2531011) & MY_RAND_MAX_32) >> 16;
	
	float p = float(r)/float(MY_RAND_MAX+1);
	
	return int(p*n);
}

// #define RNG_DEF int& rx, int& ry, int& rz, int& rw
// #define RNG_ARGS rx,ry,rz,rw
// a random number 0..(n-1), using the stored locations for x,y,z,q
// __device__ __host__ int random_int(int n, RNG_DEF) {
// 	int t;
//  
// 	t = rx ^ ( rx << 11);
// 	rx = ry; ry = rz; rz = rw;
// 	rw = rw ^ (rw >> 19) ^ (t ^ (t >> 8));
// 	
// 	return (rw%n);
// }

__device__ __host__ int random_flip(RNG_DEF) {
	return random_int(2, RNG_ARGS);
}

__device__ __host__ float random_float(RNG_DEF) {
	return float(random_int(1000000, RNG_ARGS)) / 1000000.0;
}

__device__ __host__ float random_exponential(float r, RNG_DEF) {
	return -log(random_float(RNG_ARGS))/r;
}

// Generate a 0-mean random deviate with mean 0, sd 1
// This uses Box-muller so we don't need branching
__device__ __host__ float random_normal(RNG_DEF) {
	float u = random_float(RNG_ARGS);
	float v = random_float(RNG_ARGS);
	return sqrt(-2.0*log(u)) * sin(2*PIf*v);
}

__device__ __host__ float random_lnormal(float u, float s, RNG_DEF) {
	return exp(u+s*random_normal(RNG_ARGS));
}


// geometric sample from P on 0,1,2,..., with a max (inclusive) value of m
__device__ __host__ int truncated_geometric(float P, int m, RNG_DEF){ 
	
	int i=0;
	while( random_float(RNG_ARGS) < P && i < m) i++;
	return i;
}


// the log probability of our truncated geometric
__device__ __host__ float ltruncated_geometric(int x, float P, int m){ 
	// TODO: PLEASE CHECK THIS
	
	if(x < m)     return log(P) * x + log(P-1); // standard geometric
	else if(x==m) return log(P) * m; // truncation
	else          return -1.0f/0.0f;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Statistical functions
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__device__ __host__ float lexponentialpdf(float x, float r) {
	if(x < 0.0f){ return -1.0f/0.0f; }
	return log(r) - r*x;
}

__device__ __host__ float lnormalpdf( float x, float s ){
	return -(x*x)/(2.0*s*s) - 0.5 * (log(2.0*PIf) + 2.0*log(s));
}

// log  log-normal
__device__ __host__ float llnormalpdf(float x, float s) {
	if(x <= 0) { return -1.0f/0.0f; }
	return lnormalpdf( log(x), s) - log(x);
}

__device__ __host__ float luniformpdf( float x ){
	if(x<0.0f || x>1.0f){ return -1.0f/0.0f; }
	return 0.0f;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Numerics
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

double logaddexp(double x, double y) {
	double m = max(x,y);
	return log(exp(x-m) + exp(y-m)) + m;
}

double logsumexp( double* ar, int len) {

	double m = -1./0.;
	for(int i=0;i<len;i++) {
		if(ar[i] > m) m = ar[i];
	}
	
	double thesum = 0.0;
	for(int i=0;i<len;i++) {
		thesum += exp(ar[i]-m);
	}
	
	return log(thesum)+m;
}

