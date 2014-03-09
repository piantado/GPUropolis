/*
	Assorted CUDA utilities
*/

 const float PIf = 3.141592653589;
// this is our arithmetic if/then macro
#define ifthen(x,y,z) ((y)*(x) + (z)*(1-(x)))

// Check if something is not a nan or inf
#define is_valid(x) (!(isnan(x) || isinf(x)))

// So we can use our own code in debugging
 float mul(float x, float y) { return x*y; }
 float div(float x, float y) { return x/y; }
 float add(float x, float y) { return x+y; }
 float sub(float x, float y) { return x-y; }

// Define -log(x) or log(1/x)
#define nlog(x)  (-log(x))
#define nlogf(x) (-log(float(x)))
#define logf(x)  log(float(x))

// swap two pointers conditionally, without branching
 void ifthenswap(int Q, void** x, void** y, void** tmp) {
	int i = int(Q>0);
	tmp[0] = *x;
	tmp[1] = *y;
	void* t1 = tmp[i];
	void* t2 = tmp[1-i];
	(*x) = t1;
	(*y) = t2;
}

// a random number 0..(n-1), using the stored locations for x,y,z,q
 int random_int(int n, int& x, int& y, int& z, int& w) {
	int t;
 
	t = x ^ ( x << 11);
	x = y; y = z; z = w;
	w = w ^ (w >> 19) ^ (t ^ (t >> 8));
	
	return (w%n);
}

// And some macros to help with random numbers, so we can concisely pass these around functions etc
#define RNG_DEF int& rx, int& ry, int& rz, int& rw
#define RNG_ARGS rx,ry,rz,rw

 int random_flip(int& x, int& y, int& z, int& w) {
	return random_int(2, x,y,z,w);
}

 float random_float(int& x, int& y, int& z, int& w) {
	return float(random_int(1000000, x,y,z,w)) / 1000000.0;
}

 float random_exponential(float r, int& x, int& y, int& z, int& w) {
	return -log(random_float(x,y,z,w))/r;
}

 float lexponentialpdf(float x, float r) {
	return log(r) - r*x;
}

 float lnormalpdf( float x, float s ){
	return -(x*x)/(2.0*s*s) - 0.5 * log(2.0*PIf*s*s);
}

// Generate a 0-mean random deviate with mean 0, sd 1
// This uses Box-muller so we don't need branching
 float random_normal(int& x, int& y, int& z, int& w) {
	float u = random_float(x,y,z,w);
	float v = random_float(x,y,z,w);
	return sqrt(-2.0*log(u)) * sin(2*PIf*v);
}

 float random_lnormal(float u, float s, int& x, int& y, int& z, int& w) {
	return exp(u+s*random_normal(x,y,z,w));
}

// Log normal
 float llnormalpdf(float x, float s) {
	return lnormalpdf( log(x), s) - log(x);
}
