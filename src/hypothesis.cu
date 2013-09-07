/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Hypothesis definitions and functions
 */

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Functions for setting and manipulating the program length
// here all hypotheses are allocated the maximum amount of memory
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

const int     MAX_MAX_PROGRAM_LENGTH = 200; // the most this could ever be. 
int              hMAX_PROGRAM_LENGTH = 25; // on the hostA
__constant__ int dMAX_PROGRAM_LENGTH = 25; // on the device

// we must use a function to set the program length, because its used on local and global vars
void set_MAX_PROGRAM_LENGTH(int v){
	assert(v < MAX_MAX_PROGRAM_LENGTH);
	hMAX_PROGRAM_LENGTH = v;
	cudaMemcpyToSymbol(dMAX_PROGRAM_LENGTH,&hMAX_PROGRAM_LENGTH,sizeof(int));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Hypothesis
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// what type is the program?
typedef char op_t;

// A struct for storing a hypothesis, which is a program and some accoutrements
typedef struct hypothesis {
	float prior;
	float likelihood;
	int check0;
	float posterior;
	int check1;
	float temperature;
	float proposal_generation_lp; // with what log probability does our proposal mechanism generate this? 
	int program_length; // how long is my program?
	float acceptance_ratio; // after we run, what's the acceptance ratio for each thread?
	int check2;
	op_t   program[MAX_MAX_PROGRAM_LENGTH];
	int check3;
	int chain_index; // what index am I?
// 	int mutable_end; // what is the last index of program that proposals can change?
} hypothesis;

// A standard initialization that sets this thing up!
void initialize(hypothesis* h){
// 	h->mutable_end = hMAX_PROGRAM_LENGTH-1;
	h->posterior = -1.0/0.0;
	h->prior = -1.0/0.0;
	h->likelihood = 0.0;
	h->proposal_generation_lp = 0.0; 
	h->program_length = 0.0;
	h->acceptance_ratio = 0.0;

	// Set some check bits to catch buffer overruns. If any change, we have a problem!
	h->check0 = 33;	h->check1 = 33;	h->check2 = 33;	h->check3 = 33;
	
	// zero the program
	for(int i=0;i<hMAX_PROGRAM_LENGTH;i++)
		h->program[i] = 0x0; 
}

// so we qsort so best is LAST
int hypothesis_posterior_compare(const void* a, const void* b) {
	float ap = ((hypothesis*)a)->posterior;
	float bp = ((hypothesis*)b)->posterior; 
	
	if( ap>bp ) { return 1; }
	if( ap<bp ) { return -1; }
	return 0;
}

// so we qsort so best is FIRST
int neghypothesis_posterior_compare(const void* a, const void* b) {
	return -hypothesis_posterior_compare(a,b);
}

// sort so that we can remove duplicate hypotheses
int sort_bestfirst_unique(const void* a, const void* b) {
	int c = neghypothesis_posterior_compare(a,b);
	
	if(c != 0) return c;
	else { // we must sort by the program (otherwise hyps with identical posteriors may not be removed as duplicates)
		hypothesis* ah = (hypothesis*) a;
		hypothesis* bh = (hypothesis*) b;
		for(int i=hMAX_PROGRAM_LENGTH-1;i>0;i--){
			op_t x = ah->program[i]; op_t y = bh->program[i];
			if(x<y) return 1;
			if(x>y) return -1;
		}
		
	}
	
	return 0;
}



int hypothesis_structurally_identical( hypothesis* a, hypothesis* b) {
	// check if two hypotheses are equal in terms of the program (ignoring constants and other params)
	
	if(a->program_length != b->program_length) return 0;
	
	// now loop back, only up to program_length
	for(int i=hMAX_PROGRAM_LENGTH-1;i>=hMAX_PROGRAM_LENGTH-a->program_length;i--) {
		if(a->program[i] != b->program[i]) return 0;
	}
	return 1;
}

// void print_program_as_expression(hypothesis* h); // needed by print but defined in programs.cu
// void print_hypothesis(FILE* fp, int outer, int rank, int print_stack, hypothesis* h){
// 
// 	fprintf(fp, "%d\t%d\t%d\t%f\t%4.4f\t%4.4f\t%4.4f\t", ismap, outer, rank, h->chain_index, h->posterior,  h->prior, h->likelihood, h->acceptance_ratio);
// 	
// 	if(print_stack){
// 		printf("\"");
// 		for(int i=0;i<MAX_PROGRAM_LENGTH;i++) 
// 			printf("%d ", h->program[i]);
// 		printf("\"\t");
// 	}
// 	
// 	printf("\"");
// 	print_program_as_expression( h );
// 	printf("\"\n");
// }