/*

*/

const int MAX_PROGRAM_LENGTH = 25;

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
	op_t   program[MAX_PROGRAM_LENGTH];
	int check3;
	int chain_index; // what index am I?
} hypothesis;

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

int hypothesis_structurally_identical( hypothesis* a, hypothesis* b) {
	// check if two hypotheses are equal in terms of the program (ignoring constants and other params)
	for(int i=MAX_PROGRAM_LENGTH;i>=0;i--) {
		if(a->program[i] != b->program[i]) return 0;
	}
	return 1;
}

void print_program_as_expression(hypothesis* h); // needed by print but defined in programs.cu
void print_hypothesis(int outer, int rank, int ismap, int print_stack, hypothesis* h){

	printf("%d\t%d\t%d\t%d\t%f\t%4.4f\t%4.4f\t%4.4f\t", ismap, outer, rank, h->chain_index, h->posterior,  h->prior, h->likelihood, h->acceptance_ratio);
	
	if(print_stack){
		printf("\"");
		for(int i=0;i<MAX_PROGRAM_LENGTH;i++) 
			printf("%d ", h->program[i]);
		printf("\"\t");
	}
	
	printf("\"");
	print_program_as_expression( h );
	printf("\"\n");
}