/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Functions for programs. This includes primitives, but these will be migrated to primitives.cu 
 */

// arrays to store the prior on the host and device
const int MAX_NUM_OPS = 100;
float            hPRIOR[MAX_NUM_OPS]; 
__device__ float dPRIOR[MAX_NUM_OPS];

// lower two bits give the number of arguments
#define nargs(x) NARGS[x]
#define stack_change(x) (1-nargs(x))

// how much does each op change the stack?
// #define stack_change(x) (1-NARGS[x]) 

// Choose an op from the prior distribution
// NOTE: assumes prior on expansions is normalized
__device__ op_t random_op(RNG_DEF) {
// 	float f = random_float(x,y,z,w);
// 	op_t ret = 0;
// 	int notdone = 1;
// 	for(int i=0;i<NUM_OPS;i++) {
// 		f = f-PRIOR[i];
// 		ret = ret + i*(f<=0.0)*notdone;
// 		notdone = (f>=0.0);
// 	}
// 	return ret;
	
	
	// Version with breaks. Faster, apparently. 
	float f = random_float(RNG_ARGS);
	for(int i=0;i<NUM_OPS;i++) {
		f = f-dPRIOR[i];
		if(f <= 0.0 && i != NOOP_) return i;
	}
	assert(0); // should not get here
	return NOOP_;// to prevent warning
	
}

// Starting at i in buf, go back to find the location of the match
// returns -1 for things that run off the end (no match)
__device__ int find_close_backwards(op_t* buf, int pos) {
	
	// Do a version with branching
	int nopen = -1;
	int i = pos;
	for(;i>=0;i--) {
		nopen += stack_change(buf[i]);
		if(nopen == 0) break;
	}
	
	return i;
	
}

// from start...end copy into to[dMAX_PROGRAM_LENGTH-(end-start+1)]..to[dMAX_PROGRAM_LENGTH-1]
// TODO: CHECK OFF BY 1s
// It does this while computing the close backwards
__device__ void copy_subnode_to(hypothesis* from, hypothesis* to, int pos) {
	
	int j=dMAX_PROGRAM_LENGTH-1;
	int nopen = -1;
	for(int i=pos;nopen != 0 && i>=0;i--) {
		op_t op = from->program[i];
		nopen += stack_change(op);
		
		to->program[j--] = op;
	}
	
	// update the length
	to->program_length = dMAX_PROGRAM_LENGTH - (j+1);	
}

// starting with startpos and moving backwards, find tofind in findin
__device__ int count_identical_to(hypothesis* tofind, hypothesis* findin, int startpos) {
	
	int cnt = 0;
	for(int i=startpos; i>=dMAX_PROGRAM_LENGTH-findin->program_length;i--) {
		
		int keep = 1;
		for(int j=0;j<tofind->program_length && keep;j++) {
			keep = keep && (tofind->program[dMAX_PROGRAM_LENGTH-1-j] == findin->program[i-j]);
		}
		cnt += keep;
	}
	return cnt;
	
}


// Replace pos in from with x
// NOTE: We require that from != to
// __device__ void replace_subnode_with(hypothesis* from, hypothesis* to, int pos, hypothesis* x) {
// 	int start_ar = dMAX_PROGRAM_LENGTH - from->program_length;
// 	int xstart = dMAX_PROGRAM_LENGTH - x->program_length;
// 	
// 	// set ar[start_ar...(start_ar+len_ar)] = x[MAX_PROGRAM_LENGTH-len_x:MAX_PROGRAM_LENGTH-1];
// 	// nonincludsive of end_ar, but inclusive of start_ar. inclusive of start_x
// 	// NOTE: WE can get garbage on the left of our string if we insert soemthing in 0th (rightmost) position
// 
// 	// correct for the mis-alignment of x and the gap of what its replacing
// 	int shift = x->program_length - (pos+1-start_ar); 
// 	
// 	int xi = xstart;
// 	for(int ari=0;ari<dMAX_PROGRAM_LENGTH;ari++) {
// 		int in_splice_region = (ari>=start_ar-shift) && (ari<=pos);
// 		int in_final_region = (ari > pos);
// 		
// 		// wrap this for in case it goes off the end
// 		int ar_ari_shift = ifthen( (ari+shift < 0) || (ari+shift >= dMAX_PROGRAM_LENGTH), 0, from->program[ari+shift]);
// 		
// 		to->program[ari] = ifthen(in_splice_region, x->program[xi], ifthen(in_final_region, from->program[ari], ar_ari_shift) );
// 		
// 		xi += in_splice_region; // when in the splice region, increment by 1
// 	}
// 	
// 	// and update the length! This is necessary!
// 	to->program_length = from->program_length + shift;
// 	
// }
//
// 000000XXX00
// 0000000XX00

__device__ void replace_subnode_with(hypothesis* from, hypothesis* to, int endsplice, hypothesis* x) {
	/*
	 * TODO: THere is still a weird corner case where this produces NOOP_s where we don't want. CANNOT add the assertion below.
	 * ALSO We seem to be crashing sometimes....
	 * 
	 */
	int startsplice = find_close_backwards(from->program, endsplice); 
	int xclose = dMAX_PROGRAM_LENGTH - x->program_length;
	int shift = x->program_length - (endsplice-startsplice+1);

// 	for(int i=dMAX_PROGRAM_LENGTH-from->program_length;i<dMAX_PROGRAM_LENGTH;i++){
// 		cuPrintf("\tf %d %d %d %d %d %d\n", i, from->program[i], from->program_length, x->program_length, startsplice, endsplice);
// 	}
// 	for(int i=dMAX_PROGRAM_LENGTH-x->program_length;i<dMAX_PROGRAM_LENGTH;i++){
// 		cuPrintf("\tx %d %d %d %d\n", i, x->program_length, x->program[i], 0);
// 	}	
	
	// under these situations we cannot insert
	if(from->program_length+shift < dMAX_PROGRAM_LENGTH && from->program_length+shift >= 0 && startsplice >= 0 && xclose >= 0) {
		
		
		int xi = xclose;
		for(int i=0;i<dMAX_PROGRAM_LENGTH;i++) {
			if(i+shift < 0) {
				continue; // do nothing here
			}
			else if(i+shift < startsplice) { 
				assert(i+shift < dMAX_PROGRAM_LENGTH);
				assert(i+shift>=0);
				to->program[i] = from->program[i+shift];
			}
			else if( i+shift >= startsplice && i <= endsplice){
				assert(i < dMAX_PROGRAM_LENGTH);
				assert(i>=0);
				assert(xi < dMAX_PROGRAM_LENGTH);
				assert(xi >= xclose);
				to->program[i] = x->program[xi];
				xi++;
				
			}
			else {
				assert(from->program[i] != NOOP_);
				to->program[i] = from->program[i];
			}
		}
	
		to->program_length = from->program_length + shift;
	}
	else {
		// We can't insert, so make it to nothing
		for(int i=0;i<dMAX_PROGRAM_LENGTH;i++) 
			to->program[i] = NOOP_; // from->program[i];
		
		to->program_length = 0; // from->program_length;
	}
	
}

__device__ void replace_random_subnode_with(hypothesis* from, hypothesis* to, hypothesis* x, RNG_DEF) {
	int pos =  dMAX_PROGRAM_LENGTH-1 - random_int(from->program_length, RNG_ARGS);
	replace_subnode_with(from, to, pos, x);
}


// Sets hypothesis->program_length and hypothesis->proposal_generation_lp
// This way we loop over the program one time fewer
// also the number of constants
__device__ void update_hypothesis(hypothesis* h){
	
	float gp = 0.0;
	int nopen = -1;
	int pos; 
	int nconstants = 0;
	int keep = 1; // this stores whether or not we see an erroneous NOOP_, in which case we make it infinitely bad
	
	// else actually compute	
	for(pos=dMAX_PROGRAM_LENGTH-1;;pos--) {
		
		// it will never work out
		if(pos < 0){keep = 0; break; }
		
		op_t pi = h->program[pos];
		
		// definitely should have no NOOP_s in here
		if(pi == NOOP_) { keep = 0; break; }
		
		nopen  += stack_change(pi);
		gp     += log(dPRIOR[pi]); // prior for this op
		
		nconstants += (pi == CONSTANT_);
		
		if(nopen == 0) break;
	}

		
	if(keep) {
		h->nconstants = min(nconstants,MAX_CONSTANTS); // must ensure that we don't ever think we have too many!
		h->program_length = dMAX_PROGRAM_LENGTH-pos;
		h->proposal_generation_lp = gp;
	} 
	else {
		h->nconstants = 0;
		h->program_length = 0;
		h->proposal_generation_lp = -1./0.;
	}
}


// Puts a random closed expression into buf, pushed to the rhs
// and returns the start position of the expression. 
// NOTE: Due to size constraints, this may fail, which is okay, it will 
// just create a garbage proposal
__device__ void random_closed_expression(hypothesis* to, RNG_DEF) {
	
	int nopen = -1; // to begin, we are looking for 1 arg to the left
	int len = 0;
	for(int i=dMAX_PROGRAM_LENGTH-1;i>=0;i--) {
		if(nopen != 0) {
			op_t newop = random_op(RNG_ARGS); //
			assert(newop != NOOP_);
			
			to->program[i] = newop; 
			nopen += stack_change(newop); 
			len++;
		}
		else {
			to->program[i] = NOOP_;
		}
	}
	
	to->program_length = len;
}



__device__ int find_program_close(op_t* program){
	// Where does this progrem end? (going backwards)
	return find_close_backwards(program, dMAX_PROGRAM_LENGTH-1);
}


// starting at the RHS, go until we find the "Effective" program length (ignoring things that are not used)
__device__ int find_program_length(op_t* program) {
	// how long is this program?
	return dMAX_PROGRAM_LENGTH-find_program_close(program);
}

__device__ float compute_x1depth_prior(hypothesis* h) {
	/*
	 * Extra penalty for xs that occur deeply nested. This returns the sum of X_PENALTY + X_DEPTH_PENALTY*depth  
	 * over entire tree (add it to lp_generation probability). 
	 * NOTE TODO: In the future, we should sum up the 1-these so we get a correctly normalized probability distribution
	 * 
	 * NOTE: This uses prior_stack to store the depths as we iterate through the program
	 */
	
	int prior_stack[MAX_MAX_PROGRAM_LENGTH];
	
	float mysum = 0.0;
	int close = find_program_close(h->program);
	
	int di = 1; // what depth are we at?
	prior_stack[0] = 0;
	
	for(int i=dMAX_PROGRAM_LENGTH-1;i>=close;i--) { // start at the end so we measure depth correctly
		op_t pi = h->program[i];
		
		// update our array keeping track of depth, and penalize if we should
		switch( nargs(pi) ) {
			case 0: 
				di--;
				mysum -= (pi == X_) * ( X_PENALTY + X_DEPTH_PENALTY * prior_stack[di] );
				break;
			case 1: 
				prior_stack[di] = prior_stack[di-1]+1; // add one function, incrementing our count of functions
				di++;
				break;
			case 2:
				int v = prior_stack[di-1]+1; // add one function fo two arguments
				prior_stack[di] = v;
				prior_stack[di+1] = v;
				di += 2;
				break;
		}
	}
	return mysum; // */
}


//------------
// For easier displays
const int MAX_OP_LENGTH = 256; // how much does each string add at most?
char SS[MAX_MAX_PROGRAM_LENGTH*2][MAX_OP_LENGTH*MAX_MAX_PROGRAM_LENGTH]; 

void print_program_as_expression(FILE* fp, hypothesis* h) {
	/*
	 * Print things nicely on displays.
	 * Currently these are set to play nice with sympy for processing output
	 */
	
	char buf[MAX_MAX_PROGRAM_LENGTH*MAX_MAX_PROGRAM_LENGTH];
	
	int top = MAX_MAX_PROGRAM_LENGTH; // top of the stack
	int constant_i = 0; // what constant are we pointing at?
	int program_start = hMAX_PROGRAM_LENGTH-h->program_length;
	
	// re-initialize our buffer
	for(int r=0;r<MAX_MAX_PROGRAM_LENGTH*2;r++) strcpy(SS[r], "0"); // since everything initializes to 0
	
	for(int p=program_start;p<hMAX_PROGRAM_LENGTH;p++) { // NEED to start at the same place as virtual-machine (either 0 or program_start), or else consts can get out of sync!
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
			case CONSTANT_:
				top += 1;			
				sprintf(SS[top],"%f",  (constant_i < MAX_CONSTANTS) * h->constants[constant_i]);
				constant_i += (constant_i < MAX_CONSTANTS);
				break;
			default: // Defaultly just use the name
				strcpy(buf, NAMES[op]);
				if(hNARGS[op]>=1){
					strcat(buf, "(");
					strcat(buf, SS[top]);
					for(int k=1;k<hNARGS[op];k++) { // append on all the arguments
						strcat(buf, ",");
						strcat(buf, SS[top-k]);
					}
					strcat(buf, ")");
				}
				else { // if we are a constant, stack increases
					top++;
				}
				strcpy(SS[top], buf);
				break;
				
		}
	}
	
	fprintf(fp, "%s", SS[top]);
}


