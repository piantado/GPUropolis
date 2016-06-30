/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Hypothesis definitions and functions
 */


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Set program length and number of constants
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum CONSTANT_TYPES { GAUSSIAN, EXPONENTIAL, UNIFORM,  LOGNORMAL, CAUCHY, __N_CONSTANT_TYPES};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define COPY_HYPOTHESIS(x,y) memcpy( (void*)(x), (void*)(y), sizeof(hypothesis));

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Bayes on hypotheses
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// Choose an op from the prior distribution
// NOTE: assumes prior on expansions is normalized
__device__ __host__ op_t random_op(RNG_DEF) {
    return NUM_OPS * random_float(RNG_ARGS); // TODO: CHECK 
}

// Randomize the constants
__device__  __host__ float randomize_constants(hypothesis* h, RNG_DEF) {
    for(int k=0;k<MAX_CONSTANTS;k++) {
        h->constants[k] = random_normal(RNG_ARGS);
        h->constant_types[k] = random_int(__N_CONSTANT_TYPES, RNG_ARGS);
    }
    return 0.0;
}

// These are used below but defined for real elsewhere
__device__ float f_output(float x, hypothesis* h, float* stack);


__device__ float compute_likelihood(int DLEN, datum* device_data, hypothesis* h, float* stack) {

	float ll = 0.0;
	for(int i=0;i<DLEN;i++){
		//__syncthreads(); // NOT a speed improvement
		
		float val = f_output( device_data[i].input, h, stack);
		
		// compute the difference between the output and what we see
		data_t d = device_data[i].output - val;
		
		// and use a gaussian likelihood
		ll += lnormalpdf(d, device_data[i].sd);
	}
	
	// ensure valid
	if (!is_valid(ll)) ll = -1.0f/0.0f;
	
 	h->likelihood = ll / LL_TEMPERATURE;
	
	return h->likelihood;
	
}


__device__ __host__ int count_constants(hypothesis* h){
    // how many constants are used?
    int cnt = 0;
    for(int i=0;i<MAX_PROGRAM_LENGTH;i++){
        op_t op = h->program[i];
        if(op == CONSTANT_) cnt++;
    }
    return cnt;
}

// Compute the prior on constats
// assuming constant types are generated uniformly
__device__ float compute_constants_prior(hypothesis* h) {
    
    float lp = 0.0;
    
//     int mx = count_constants(h); // only penalize in the prior for as many as we use
//     for(int k=0;k<min(mx, MAX_CONSTANTS);k++) {
    
    for(int k=0;k<MAX_CONSTANTS;k++) {
        float value = h->constants[k];
        switch(h->constant_types[k]) {
            case GAUSSIAN:    lp += lnormalpdf(value, 1.0); break;
            case LOGNORMAL:   lp += llnormalpdf(value, 1.0); break;
            case CAUCHY:      lp += lcauchypdf(value, 1.0); break;
            case EXPONENTIAL: lp += lexponentialpdf(value, 1.0); break;
            case UNIFORM:     lp += luniformpdf(value); break;  
        }
    }
    
    return lp;
}



__device__ __host__ int compute_program_length(hypothesis* h){
    // how many non-NOOP_ operations are there until the end?
    int cnt = 0;
    for(int i=0;i<MAX_PROGRAM_LENGTH;i++){
        op_t op = h->program[i];
        
        if(op != NOOP_) cnt++;
        
        if(op == RET_) break;
    }
    return cnt;
}

__device__ float compute_prior(hypothesis* h) {
    
    // We'll use just a simple exp(-length)
    h->prior = (-float(compute_program_length(h)) + compute_constants_prior(h)) / PRIOR_TEMPERATURE;   
    
    return h->prior;
}   

__device__ void compute_posterior(int DLEN, datum* device_data, hypothesis* h, float* stack) {
	compute_prior(h);
	compute_likelihood(DLEN, device_data, h, stack);
        
	h->posterior = h->prior + h->likelihood;
}



// A standard initialization that sets this thing up!
void initialize(hypothesis* h, RNG_DEF){
    h->posterior = -1.0/0.0;
    h->prior = -1.0/0.0;
    h->likelihood = 0.0;
    
    // zero the program
    for(int i=0;i<MAX_PROGRAM_LENGTH;i++)
        h->program[i] = random_op(RNG_ARGS); 
    
    // Set up our constants -- just random
    randomize_constants(h, RNG_ARGS);
}

// so we qsort so best is LAST
int hypothesis_posterior_compare(const void* a, const void* b) {
    float ap = ((hypothesis*)a)->posterior;
    float bp = ((hypothesis*)b)->posterior; 
    
    // use != here to check for nan:
    if( ap>bp || (bp!=bp) ) { return 1; }
    if( ap<bp || (ap!=ap) ) { return -1; }
    return 0;
}



const int MAX_OP_LENGTH = 5; // how much does each string add at most?
char buff[100]; // a buffer to store floating points

void print_program_as_expression(FILE* fp, hypothesis* h) {
    // Display a program in a file fp
    
    int L = MAX_OP_LENGTH*MAX_PROGRAM_LENGTH; // max length 
    char buf[L];
    strcpy(buf, "");
        
    for(int p=0;p<MAX_PROGRAM_LENGTH;p++){
        switch(h->program[p]) {
            case NOOP_: strcat(buf, "NOOP "); break;
            case ZERO_: strcat(buf, "0 "); break;
            case ONE_:  strcat(buf, "1 "); break;
            case X_:    strcat(buf, "x "); break;
            case ADD_:  strcat(buf, "+ "); break;
            case SUB_:  strcat(buf, "- "); break;
            case MUL_:  strcat(buf, "* "); break;
            case DIV_:  strcat(buf, "/ "); break;
            case POW_:  strcat(buf, "^ "); break;
            case NEG_:  strcat(buf, "~ "); break;
            
            case LOG_:  strcat(buf, "log "); break;
            case EXP_:  strcat(buf, "exp "); break;
            
            case DUP_:  strcat(buf, "DUP "); break;
            case SWAP_: strcat(buf, "SWAP "); break;
            case RET_:
                strcat(buf, "ret");
                goto DONE;            
            case CONSTANT_:  
                strcat(buf, "C ");
                /* If we want to print the numerical value rather than "C"
                sprintf(buff, "%f ", h->constants[constant_i%MAX_CONSTANTS]);
                constant_i++; // If you use this, define it above
                strcat(buf, buff); */
                break;
            default:
                cerr << "ERROR unknown primitive" << int(h->program[p]) << endl;
        }
    }
DONE:

    fprintf(fp, "%s", buf);
}

void dump_to_file(FILE* fp, hypothesis* h, int i, int outern) {
    
    fprintf(fp, "%i\t%d\t%d\t%.3f\t%.3f\t%.3f\t", 
        outern, 
        i, 
        h->chain_index, 
        h->posterior,  
        h->prior,
        h->likelihood
        );
    
        //print out the program
        fprintf(fp,"\"");
        for(int i=0;i<MAX_PROGRAM_LENGTH;i++) fprintf(fp, "%d ", h->program[i]);
        fprintf(fp,"\"\t");     
        
        // print out constant types
        fprintf(fp,"\"");
        for(int i=0;i<MAX_CONSTANTS;i++) fprintf(fp, "%i ", h->constant_types[i]);
        fprintf(fp,"\"\t");

        // print out constants
        fprintf(fp,"");
        for(int i=0;i<MAX_CONSTANTS;i++) fprintf(fp, "%.5f\t", h->constants[i]);
        fprintf(fp,"\t");           
        
        fprintf(fp, "\"");
        print_program_as_expression(fp, h );
        fprintf(fp, "\"\n");
}