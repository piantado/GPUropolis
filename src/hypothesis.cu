/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Hypothesis definitions and functions
 */


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Set program length and number of constants
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

enum CONSTANT_TYPES { GAUSSIAN, EXPONENTIAL,  LOGNORMAL, CAUCHY, __N_CONSTANT_TYPES};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#define COPY_HYPOTHESIS(x,y) memcpy( (void*)(x), (void*)(y), sizeof(hypothesis));

// a bad log prob to return for NANs. -inf is finicky on CUDA
#define BAD_LP (-9e99)

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Bayes on hypotheses
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


// Choose an op from the prior distribution
// NOTE: assumes prior on expansions is normalized
__device__ __host__ op_t random_op(RNG_DEF) {
    return random_int(NUM_OPS, RNG_ARGS); // TODO: CHECK 
}

// Randomize the constants, returning their lp
__device__  __host__ void randomize_constants(hypothesis* h, RNG_DEF) {
    
    for(int k=0;k<MAX_CONSTANTS;k++) {
        h->constants[k] = random_normal(RNG_ARGS);
        h->constant_types[k] = random_int(__N_CONSTANT_TYPES, RNG_ARGS);
    }
}

// These are used below but defined for real elsewhere
__device__ float f_output(float x, hypothesis* h, float* stack);


__device__ float compute_likelihood(int DLEN, datum* device_data, hypothesis* h, float* stack) {

	float ll = 0.0;
    
	for(int i=0;i<DLEN;i++){
		//__syncthreads(); // NOT a speed improvement
		
		float val = f_output( device_data[i].input, h, stack);
		
        if(!is_valid(val)) { 
            h->likelihood = BAD_LP;
            return h->likelihood;
        }
        
		// compute the difference between the output and what we see
		data_t d = device_data[i].output - val;
		
		// and use a gaussian likelihood
		ll += lnormalpdf(d, device_data[i].sd);
        
	}
	
	// ensure valid
	if (!is_valid(ll)) { 
        h->likelihood = BAD_LP;
        return h->likelihood;
    }
	else {
        h->likelihood = ll / LL_TEMPERATURE;
		return h->likelihood;
    }
	
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
//             case UNIFORM:     lp += luniformpdf(value); break;  // this is a problem because it gives crazy values for the constants!
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
    h->posterior = BAD_LP;
    h->prior = BAD_LP;
    h->likelihood = 0.0;
    
    // randomize the program
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



const int MAX_OP_LENGTH = 10; // how much does each string add at most?
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
                cerr << "ERROR unknown primitive " << int(h->program[p]) << "\t" << p << endl;
        }
    }
DONE:

    fprintf(fp, "%s", buf);
}


//------------
// For easier displays
const int BUF_LEN = MAX_OP_LENGTH*MAX_PROGRAM_LENGTH*10; 
char SS[STACK_SIZE][BUF_LEN]; 
void print_program_as_concise_expression(FILE* fp, hypothesis* h) {
    /*
     * Print things nicely on displays.
     * Currently these are set to play nice with sympy for processing output
     */
    
    char buf[BUF_LEN];
    
    int top = STACK_START; // top of the stack
    int constant_i = 0; //
    
    // re-initialize our buffer
    for(int r=0;r<STACK_SIZE;r++) 
        strcpy(SS[r], "0"); // since everything initializes to 0
    
    for(int p=0;p<MAX_PROGRAM_LENGTH;p++) {
        op_t op = h->program[p];
        
        switch(op) {
            case NOOP_:
                break;
            case ZERO_: 
                top += 1;
                strcpy(SS[top], "0"); // TODO: Shoudl ZERO_ And ONE_ be floats? Sympy doesn't fully simplify if floats
                break;
            case ONE_: 
                top += 1;
                strcpy(SS[top], "1");
                break;
            case X_:
                top += 1;
                strcpy(SS[top], "x");
                break;
            case ADD_:
                strcpy(buf, "(");
                strcat(buf, SS[top]);
                strcat(buf, "+");
                strcat(buf, SS[top-1]);
                strcat(buf, ")");
                top -= 1;
                strcpy(SS[top], buf);
                break;
                
            case SUB_:
                strcpy(buf, "(");
                strcat(buf, SS[top]);
                strcat(buf, "-");
                strcat(buf, SS[top-1]);
                strcat(buf, ")");
                top -= 1;
                strcpy(SS[top], buf);
                break;
                
            case MUL_:
                strcpy(buf, "(");
                strcat(buf, SS[top]);
                strcat(buf, "*");
                strcat(buf, SS[top-1]);
                strcat(buf, ")");
                top -= 1;
                strcpy(SS[top], buf);
                break;
                
            case DIV_:
                strcpy(buf, "(");
                strcat(buf, SS[top]);
                strcat(buf, "/");
                strcat(buf, SS[top-1]);
                strcat(buf, ")");
                top -= 1;
                strcpy(SS[top], buf);
                break;
            case POW_:
                strcpy(buf, "(");
                strcat(buf, SS[top]);
                strcat(buf, "**");
                strcat(buf, SS[top-1]);
                strcat(buf, ")");
                top -= 1;
                strcpy(SS[top], buf);
                break;  
            case NEG_:
                strcpy(buf, "(-");
                strcat(buf, SS[top]);
                strcat(buf, ")");
                strcpy(SS[top], buf);
                break;
            case EXP_:
                strcpy(buf, "exp(");
                strcat(buf, SS[top]);
                strcat(buf, ")");
                strcpy(SS[top], buf);
                break;
            case LOG_:
                strcpy(buf, "log(");
                strcat(buf, SS[top]);
                strcat(buf, ")");
                strcpy(SS[top], buf);
                break;
            case CONSTANT_:
                top += 1;           
                sprintf(SS[top],"C%i", constant_i % MAX_CONSTANTS);
                constant_i++;
                break;
            case DUP_:
                top += 1;
                strcpy(SS[top], SS[top-1]);                
                break;
            case SWAP_:
                strcpy(buf, SS[top-1]);
                strcpy(SS[top-1], SS[top]);
                strcpy(SS[top], buf);                                
                break;
            case RET_:
                goto DONE;
            default: // Defaultly just use the name
                cerr << "Error in displaying ... " << int(op) << endl;
                break;
                
        }
    }
    DONE:
    
    fprintf(fp, "%s", SS[top]);
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
//         fprintf(fp,"\"");
//         for(int i=0;i<MAX_PROGRAM_LENGTH;i++) fprintf(fp, "%d-", h->program[i]);
//         fprintf(fp,"\"\t");     
//         
        fprintf(fp, "%i\t", compute_program_length(h));
        
        // print out constant types
        for(int i=0;i<MAX_CONSTANTS;i++) fprintf(fp, "%i\t", h->constant_types[i]);
        
        // print out constants
        for(int i=0;i<MAX_CONSTANTS;i++) fprintf(fp, "%.5f\t", h->constants[i]);
        
        fprintf(fp, "\"");
        print_program_as_concise_expression(fp, h);
        fprintf(fp, "\"\t");
        
        fprintf(fp, "\"");
        print_program_as_expression(fp, h );
        fprintf(fp, "\"\n");
}