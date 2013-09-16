/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Assorted misc functions
 */

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Some math
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__constant__ const float PIf = 3.141592653589;

// Check if something is not a nan or inf
#define is_valid(x) (!(isnan(x) || isinf(x)))

// Define -log(x) or log(1/x)
#define nlog(x)  (-log(x))
#define nlogf(x) (-log(float(x)))
#define logf(x)  log(float(x))

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

// a random number 0..(n-1), using the stored locations for x,y,z,q
__device__ int random_int(int n, RNG_DEF) {
	int t;
 
	t = rx ^ ( rx << 11);
	rx = ry; ry = rz; rz = rw;
	rw = rw ^ (rw >> 19) ^ (t ^ (t >> 8));
	
	return (rw%n);
}

__device__ int random_flip(RNG_DEF) {
	return random_int(2, RNG_ARGS);
}

__device__ float random_float(RNG_DEF) {
	return float(random_int(1000000, RNG_ARGS)) / 1000000.0;
}

__device__ float random_exponential(float r, RNG_DEF) {
	return -log(random_float(RNG_ARGS))/r;
}

// Generate a 0-mean random deviate with mean 0, sd 1
// This uses Box-muller so we don't need branching
__device__ float random_normal(int& x, int& y, int& z, int& w) {
	float u = random_float(x,y,z,w);
	float v = random_float(x,y,z,w);
	return sqrt(-2.0*log(u)) * sin(2*PIf*v);
}

__device__ float random_lnormal(float u, float s, int& x, int& y, int& z, int& w) {
	return exp(u+s*random_normal(x,y,z,w));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Statistical functions
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__device__ __host__ float lexponentialpdf(float x, float r) {
	return log(r) - r*x;
}

__device__ __host__ float lnormalpdf( float x, float s ){
	return -(x*x)/(2.0*s*s) - 0.5 * log(2.0*PIf*s*s);
}

// log  log-normal
__device__ __host__ float llnormalpdf(float x, float s) {
	return lnormalpdf( log(x), s) - log(x);
}

