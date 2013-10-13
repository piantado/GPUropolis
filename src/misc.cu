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

// Needed to define CUDART_NAN_F
#include "math_constants.h"

// use these since the functions isnan, ininf don't seem to work right for single precision...
// #define is_invalid(x) (isnan(x) || isinf(x))
#define is_invalid(x) (x==CUDART_INF_F || x == CUDART_NAN_F)
#define is_valid(x) (!is_invalid(x))

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Functions for avoiding branching. Most of these are just slow
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// this is our arithmetic if/then macro
#define ifthen(x,y,z) ((y)*(x) + (z)*(1-(x)))

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Random number generation 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// And some macros to help with random numbers, so we can concisely pass these around functions etc
#define RNG_DEF int& rx, int& ry, int& rz, int& rw
#define RNG_ARGS rx,ry,rz,rw
// for use on the host:


// a random number 0..(n-1), using the stored locations for x,y,z,q
__device__ __host__ int random_int(int n, RNG_DEF) {
	int t;
 
	t = rx ^ ( rx << 11);
	rx = ry; ry = rz; rz = rw;
	rw = rw ^ (rw >> 19) ^ (t ^ (t >> 8));
	
	return (rw%n);
}

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
__device__ __host__ float random_normal(int& x, int& y, int& z, int& w) {
	float u = random_float(x,y,z,w);
	float v = random_float(x,y,z,w);
	return sqrt(-2.0*log(u)) * sin(2*PIf*v);
}

__device__ __host__ float random_lnormal(float u, float s, int& x, int& y, int& z, int& w) {
	return exp(u+s*random_normal(x,y,z,w));
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
	else          return -1.0/0.0;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Statistical functions
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__device__ __host__ float lexponentialpdf(float x, float r) {
	if(x < 0){ return -1.0/0.0; }
	return log(r) - r*x;
}

__device__ __host__ float lnormalpdf( float x, float s ){
	return -(x*x)/(2.0*s*s) - 0.5 * (log(2.0*PIf) + 2.0*log(s));
}

// log  log-normal
__device__ __host__ float llnormalpdf(float x, float s) {
	if(x <= 0) { return -1.0/0.0; }
	return lnormalpdf( log(x), s) - log(x);
}

__device__ __host__ float luniformpdf( float x ){
	if(x<0 || x>1){ return -1.0/0.0; }
	return 0.0;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// My versions of primitives
// NOTE: USE OF BUILT-IN PRIMITIVES GAVE ME WEIRDO RESULTS!
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__device__ float my_pow(float x, float y) {
	if(x < 0. || is_invalid(x) || is_invalid(y)) return CUDART_NAN_F;
	if(x==0.0) return 0.0;
	else return __powf(x,y);
}

__device__ float my_exp(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return __expf(x);
}

__device__ float my_log(float x) {
	if(x < 0. || is_invalid(x) ) return CUDART_NAN_F;
	else return __logf(x);
}

__device__ float my_gamma(float x) {
	if(x < 0. ||  is_invalid(x)) return CUDART_NAN_F;
	else return tgamma(x);
}

__device__ float my_sqrt(float x) {
	if(x < 0. || is_invalid(x)) return CUDART_NAN_F;
	else return __fsqrt_rz(x);
}

__device__ float my_abs(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return abs(x);
}

__device__ float my_round(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return round(x);
}

__device__ float my_neg(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return -x;
}

__device__ float my_sgn(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return (x>0.)-(x<0.);
}

__device__ float my_add(float x, float y) {
	if(is_invalid(x) || is_invalid(y) ) return CUDART_NAN_F;
	else return __fadd_rz(x,y);
}

__device__ float my_mul(float x, float y) {
	if(is_invalid(x) || is_invalid(y) ) return CUDART_NAN_F;
	else return __fmul_rz(x,y);
}

__device__ float my_div(float x, float y) {
	if(is_invalid(x) || is_invalid(y) || y==0.0 ) return CUDART_NAN_F;
	else return __fdiv_rz(x,y);
}

__device__ float my_sin(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return __sinf(x);
}
__device__ float my_asin(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return asinf(x);
}

__device__ float my_cos(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return __cosf(x);
}
__device__ float my_acos(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return acosf(x);
}

__device__ float my_tan(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return __tanf(x);
}
__device__ float my_atan(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return atanf(x);
}