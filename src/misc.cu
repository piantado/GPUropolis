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
// Functions for avoiding branching. Most of these are just slow
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// this is our arithmetic if/then macro
#define ifthen(x,y,z) ((y)*(x) + (z)*(1-(x)))

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Random number generation 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// And some macros to help with random numbers, so we can concisely pass these around functions etc
// #define RNG_DEF int& rx, int& ry, int& rz, int& rw
// #define RNG_ARGS rx,ry,rz,rw
// 
#define RNG_DEF int& rx
#define RNG_ARGS rx
#define MY_RAND_MAX ((1U << 15) - 1)
// for use on the host:

// A linear congruential 
 __device__ __host__ int random_int(int n, RNG_DEF) {
     // number in [0,(n-1)]

	//http://rosettacode.org/wiki/Linear_congruential_generator#C (WOW, error on that page -- should be % not &)
	rx = (rx * 1103515245 + 12345) % MY_RAND_MAX;
	
	float p = float(rx)/float(MY_RAND_MAX);
	
	return int(p*n);
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

__device__ __host__ float lexponentialpdf(float x, float r) { // r exp(-r x)
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

__device__ __host__ float lcauchypdf( float x, float s ){
    return -log(PIf) - log(s) - log(1+x*x/s);
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// My versions of primitives
// Here, we use the full precision (powf instead of __powf) BUT we use the fast_math flag in the Makefile to 
// compile these to faster versions
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__device__ float my_pow(float x, float y) {
	// 	
// 	// HMM CUDA: if I write this as this, it doesn't work and I get assertion failures
// // 	if(x<0.0f) return CUDART_NAN_F;
// 	assert(x>=0.0f); // THIS FAILS WTF
// 	return powf(x,y);	
// 	// is this something to do with thread divergence?
// 	
// 	// Like this, it seems to go okay. Why?? CUDA, please explain!
	if(x>=0.0f && is_valid(x) && is_valid(y) ){
		assert(x>=0.0f);
		return powf(x,y);
	}
	else {
		return CUDART_NAN_F;
	}
}

__device__ float my_exp(float x) {
	if(is_valid(x)){
		return expf(x);
	}
	else {
		return CUDART_NAN_F;
	}
}

__device__ float my_log(float x) {
	
	if(is_valid(x) && x>0.0f) {
		return logf(x);
	}
	else {
		return CUDART_NAN_F;
	}
}

__device__ float my_gamma(float x) {
	if(x > 0.0 && is_valid(x)) {
		return tgammaf(x);
	}
	else {
		return CUDART_NAN_F;
	}
}

__device__ float my_sqrt(float x) {
	if(x < 0.0f || is_invalid(x)) return CUDART_NAN_F;
	else return sqrtf(x);
}

__device__ float my_abs(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return fabsf(x);
}

__device__ float my_round(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return roundf(x);
}

__device__ float my_neg(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return -x;
}

__device__ float my_sgn(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return (x>0.0f)-(x<0.0f);
}

__device__ float my_add(float x, float y) {
	if(is_invalid(x) || is_invalid(y) ) return CUDART_NAN_F;
	else return x+y;
}

__device__ float my_mul(float x, float y) {
	if(is_invalid(x) || is_invalid(y) ) return CUDART_NAN_F;
	else return x*y;
}

__device__ float my_div(float x, float y) {
	if(is_invalid(x) || is_invalid(y) || y==0.0f ) return CUDART_NAN_F;
	else return x/y;
}

__device__ float my_sin(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return sinf(x);
}
__device__ float my_asin(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return asinf(x);
}

__device__ float my_cos(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return cosf(x);
}
__device__ float my_acos(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return acosf(x);
}

__device__ float my_tan(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return tanf(x);
}
__device__ float my_atan(float x) {
	if(is_invalid(x)) return CUDART_NAN_F;
	else return atanf(x);
}