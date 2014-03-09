/*

*/

// CONSTANTS, 2-ARY, 1-ARY
// NUM_OPS stores the number of opsn
// TODO: IF YOU CHANGE THESE, MYOU MSUT CHANGE THE CONDITION BELOW FOR DECODING WHAT HAPPENS TO THE TOP OF STACK
/// NOOP must be first (=0) b/c we use that in the VM to keep thing from overrunning the stack
// NOTE: Things slow down if we make these  since they are not accessed at the same time
enum OPS                               { NOOP, X, ONE,      ADD, SUB, MUL, DIV, POW,    NEG, LOG, EXP, SIN, ASIN, NUM_OPS };
 int NARGS[NUM_OPS]      = { 0, 0,   0,          2,   2,   2,   2,   2,      1,  1,  1,    1,     1}; // how many args for each op?

/*
 * The expected length satisfies:
 * E = pconst + p1arg(E+1) + p2arg(2 E + 1)
 * 
 * so
 * 
 * E = 1/(1-p1arg - 2 p2arg)
 * 
 * Constraining p1arg = p2arg,
 * 
 * E = 1/(1-3p1arg)
 * 
 * so
 * 
 * p1arg = p2arg = (1-1/E)/3
 */
const float EXPECTED_LENGTH = 10.0; // how long should programs be in the prior? 
const float PARG = (1.0-1.0/EXPECTED_LENGTH)/3.0; 
const float px_ = (1.0 - 2.0*PARG)*1.0/11.0; // constrain that p1 = 10*px
const float p1_ = (1.0 - 2.0*PARG)*10.0/11.0;
const float p1arg_ = PARG / 5.0; // probability mass for EACH 1-arg element
const float p2arg_ = PARG / 5.0; // probability mass for EACH 2-arg element

float prior[NUM_OPS]    = { 0.0, px_, p1_,  p2arg_, p2arg_,  p2arg_,  p2arg_,  p2arg_,     p1arg_,  p1arg_,  p1arg_,  p1arg_,  p1arg_ };
//  float prior[NUM_OPS]    = { 0.0, px_, p1_,  p2arg_, p2arg_,  p2arg_,  p2arg_,  p2arg_,     p1arg_,  p1arg_,  p1arg_,  p1arg_,  p1arg_ };
float NUM_OPSf = float(NUM_OPS);
int FIRST_1ARG = NEG; // NOTE: INSERT/DELETE requires this arrangement of: constant, 2arg, 1arg
int FIRST_2ARG = ADD;

// how much does each op change the stack?
#define stack_change(x) (1-NARGS[x]) 

// Choose an op from the prior distribution
// NOTE: assumes prior on expansions is normalized
  op_t random_op(int& x, int& y, int& z, int& w) {
// 	float f = random_float(x,y,z,w);
// 	op_t ret = 0;
// 	int notdone = 1;
// 	for(int i=0;i<NUM_OPS;i++) {
// 		f = f-prior[i];
// 		ret = ret + i*(f<=0.0)*notdone;
// 		notdone = (f>=0.0);
// 	}
// 	return ret;
	
	
	// Version with breaks. Faster, apparently. 
	float f = random_float(x,y,z,w);
	for(int i=0;i<NUM_OPS;i++) {
		f = f-prior[i];
		if(f <= 0.0) return i;
	}
	assert(0); // should not get here
	return NOOP;// to prevent warning
	
}

// Starting at i in buf, go back to find the location of the match
  int find_close_backwards(op_t* buf, int pos) {
	
	// These two options are about equally quick.
	
// 	int nopen = -1;
// 	int notdone = 0;
// 	int ret = pos+1;
// 	for(int i=MAX_PROGRAM_LENGTH-1;i>=0;i--) {
// 		notdone =  (i==pos) || (notdone && nopen != 0);
// 		nopen  +=  (i<=pos && notdone) * stack_change(buf[i]);
// 		ret    -=  (i<=pos && notdone); 
// 	}
// 	return ret;
	
	
	// Do a version with branching
	int nopen = -1;
	int i = pos;
	for(;i>=0;i--) {
		nopen += stack_change(buf[i]);
		if(nopen == 0) break;
	}
	return i;
	
}

// from start...end copy into to[MAX_PROGRAM_LENGTH-(end-start+1)]..to[MAX_PROGRAM_LENGTH-1]
// TODO: CHECK OFF BY 1s
// It does this while computing the close backwards
  void copy_subnode_to(hypothesis* from, hypothesis* to, int pos) {
	
	int j=MAX_PROGRAM_LENGTH-1;
	int nopen = -1;
	for(int i=pos;nopen != 0 && i>=0;i--) {
		op_t op = from->program[i];
		nopen += stack_change(op);
		
		to->program[j--] = op;
	}
	
	// update the length
	to->program_length = MAX_PROGRAM_LENGTH - (j+1);	
}

// starting with startpos and moving backwards, find tofind in findin
  int count_identical_to(hypothesis* tofind, hypothesis* findin, int startpos) {
	
	int cnt = 0;
	for(int i=startpos; i>=MAX_PROGRAM_LENGTH-findin->program_length;i--) {
		
		int keep = 1;
		for(int j=0;j<tofind->program_length && keep;j++) {
			keep = keep && (tofind->program[MAX_PROGRAM_LENGTH-1-j] == findin->program[i-j]);
		}
		cnt += keep;
	}
	return cnt;
	
}


// Replace pos in from with x
// NOTE: We require that from != to
  void replace_subnode_with(hypothesis* from, hypothesis* to, int pos, hypothesis* x) {
	int start_ar = MAX_PROGRAM_LENGTH - from->program_length;
	int xstart = MAX_PROGRAM_LENGTH - x->program_length;
	
	// set ar[start_ar...(start_ar+len_ar)] = x[MAX_PROGRAM_LENGTH-len_x:MAX_PROGRAM_LENGTH-1];
	// nonincludsive of end_ar, but inclusive of start_ar. inclusive of start_x
	// NOTE: WE can get garbage on the left of our string if we insert soemthing in 0th (rightmost) position

	// correct for the mis-alignment of x and the gap of what its replacing
	int shift = x->program_length - (pos+1-start_ar); 
	
	int xi = xstart;
	for(int ari=0;ari<MAX_PROGRAM_LENGTH;ari++) {
		int in_splice_region = (ari>=start_ar-shift) && (ari<=pos);
		int in_final_region = (ari > pos);
		
		// wrap this for in case it goes off the end
		int ar_ari_shift = ifthen( (ari+shift < 0) || (ari+shift >= MAX_PROGRAM_LENGTH), 0, from->program[ari+shift]);
		
		to->program[ari] = ifthen(in_splice_region, x->program[xi], ifthen(in_final_region, from->program[ari], ar_ari_shift) );
		
		xi += in_splice_region; // when in the splice region, increment by 1
	}
	
	// and update the length! This is necessary!
	to->program_length = from->program_length + shift;
	
}

  void replace_random_subnode_with(hypothesis* from, hypothesis* to, hypothesis* x, RNG_DEF) {
	int pos =  MAX_PROGRAM_LENGTH-1 - random_int(from->program_length, RNG_ARGS);
	replace_subnode_with(from, to, pos, x);
}




// Sets hypothesis->program_length and hypothesis->proposal_generation_lp
// This way we loop over the program one time fewer
  void compute_length_and_proposal_generation_lp(hypothesis* h){
	
	float gp = 0.0;
	op_t* pi_ptr = h->program + (MAX_PROGRAM_LENGTH-1);
	int nopen = -1;
	int pos=MAX_PROGRAM_LENGTH-1;
	for(;pos>=0;pos--) {
		op_t pi = *pi_ptr;
		
		nopen  += stack_change(pi);
		gp     += log(prior[pi]); // prior for this op
		
		if(nopen == 0) break;
		pi_ptr--; // move to previous program value
	}
	
	h->program_length = MAX_PROGRAM_LENGTH-pos;
	h->proposal_generation_lp = gp;

}


// Puts a random closed expression into buf, pushed to the rhs
// and returns the start position of the expression. 
// NOTE: Due to size constraints, this may fail, which is okay, it will 
// just create a garbage proposal
  void random_closed_expression(hypothesis* to, RNG_DEF) {
	
	int nopen = -1; // to begin, we are looking for 1 arg to the left
	int len = 0;
	for(int i=MAX_PROGRAM_LENGTH-1;i>=0;i--) {
		
		if(nopen != 0) {
			op_t newop = random_op(RNG_ARGS); //
			to->program[i] = newop; 
			nopen += stack_change(newop); 
			len++;
		}
		else {
			to->program[i] = NOOP;
		}
	}
	to->program_length = len;
	
	
// 	int nopen = -1; // to begin, we are looking for 1 arg to the left
// 	int notdone = 1;
// 	int len = 0;
// 	for(int i=MAX_PROGRAM_LENGTH-1;i>=0;i--) {
// 		op_t newop = random_op(RNG_ARGS); // random_int(NUM_OPS,x,y,z,w);
// 		to->program[i] = notdone * newop; 
// 		
// 		nopen += notdone * stack_change(newop); //(1-NARGS[newop]);
// 		len += notdone;
// 
// 		notdone = notdone && (nopen != 0); // so when set done=0 when nopen=0 -- when we've reached the bottom!
// 	}
// 	to->program_length = len;
	
}



  int find_program_close(op_t* program){
	// Where does this progrem end? (going backwards)
	return find_close_backwards(program, MAX_PROGRAM_LENGTH-1);
}


// starting at the RHS, go until we find the "Effective" program length (ignoring things that are not used)
  int find_program_length(op_t* program) {
	// how long is this program?
	return MAX_PROGRAM_LENGTH-find_program_close(program);
}


// Compute the probability of generating this program according to the model!
  float compute_generation_probability(hypothesis* h) {
	/*
	// A version with uniform branching
	float lprior = 0.0;
	int close = find_program_close(h->program);
	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
		lprior     += (i>=close)*log(prior[h->program[i]]); // prior for this op
	}
	return lprior;
	/*/
	// version that doesn't loop over the entire program
	float lprior = 0.0;
	int close = find_program_close(h->program);
	for(int i=close;i<MAX_PROGRAM_LENGTH;i++) {
		lprior += log(prior[h->program[i]]); // prior for this op
	}
	return lprior; // */
}


//------------
// For easier displays
const int MAX_OP_LENGTH = 256; // how much does each string add at most?
char SS[MAX_PROGRAM_LENGTH*2][MAX_OP_LENGTH*MAX_PROGRAM_LENGTH]; 


void print_program_as_expression(hypothesis* h) {
	/*
	 * Print things nicely on displays.
	 * Currently these are set to play nice with sympy for processing output
	 */
	
	char buf[MAX_PROGRAM_LENGTH*MAX_PROGRAM_LENGTH];
	
	int top = MAX_PROGRAM_LENGTH; // top of the stack
	
	// re-initialize our buffer
	for(int r=0;r<MAX_PROGRAM_LENGTH*2;r++) strcpy(SS[r], "0"); // since everything initializes to 0
	
	for(int p=0;p<MAX_PROGRAM_LENGTH;p++) {
		op_t op = h->program[p];
		
		switch(op) {
// 			case ZERO: 
// 				top += 1;
// 				strcpy(SS[top], "0");
// 				break;
			case ONE: 
				top += 1;
				strcpy(SS[top], "1.0");
				break;
			case X:
				top += 1;
				strcpy(SS[top], "x");
				break;
// 			case PI:
// 				top += 1;
// 				strcpy(SS[top], "PI");
// 				break;
// 			case E:
// 				top += 1;
// 				strcpy(SS[top], "E");
// 				break;
			case ADD:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, ")+(");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case SUB:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, ")-(");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case MUL:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, ")*(");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case DIV:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, ")/(");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
			case POW:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, ")**(");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;	
				
			case LOG:
				strcpy(buf, "log(");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
				
			case EXP:
				strcpy(buf, "exp(");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
			case SIN:
				strcpy(buf, "sin(");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
			case ASIN:
				strcpy(buf, "asin(");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
			case NEG:
				strcpy(buf, "-");
				strcat(buf, SS[top]);
				strcat(buf, "");
				strcpy(SS[top], buf);
				break;	
// 			case TAN:
// 				strcpy(buf, "tan(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ")");
// 				strcpy(SS[top], buf);
// 				break;
// 				
		}
	}
	
	printf("%s", SS[top]);
}



// void print_program_as_expression(hypothesis* h) {
// 	/*
// 	 * Print things nicely on displays.
// 	 */
// 	
// 	char buf[MAX_PROGRAM_LENGTH*MAX_PROGRAM_LENGTH];
// 	
// 	int top = MAX_PROGRAM_LENGTH; // top of the stack
// 	
// 	// re-initialize our buffer
// 	for(int r=0;r<MAX_PROGRAM_LENGTH*2;r++) strcpy(SS[r], "0"); // since everything initializes to 0
// 	
// 	for(int p=0;p<MAX_PROGRAM_LENGTH;p++) {
// 		int op = h->program[p];
// 		
// 		switch(op) {
// // 			case ZERO: 
// // 				top += 1;
// // 				strcpy(SS[top], "0");
// // 				break;
// 			case ONE: 
// 				top += 1;
// 				strcpy(SS[top], "1.0");
// 				break;
// 			case X:
// 				top += 1;
// 				strcpy(SS[top], "x");
// 				break;
// // 			case PI:
// // 				top += 1;
// // 				strcpy(SS[top], "PI");
// // 				break;
// // 			case E:
// // 				top += 1;
// // 				strcpy(SS[top], "E");
// // 				break;
// 			case ADD:
// 				strcpy(buf, "add(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ",");
// 				strcat(buf, SS[top-1]);
// 				strcat(buf, ")");
// 				top -= 1;
// 				strcpy(SS[top], buf);
// 				break;
// 				
// 			case SUB:
// 				strcpy(buf, "sub(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ",");
// 				strcat(buf, SS[top-1]);
// 				strcat(buf, ")");
// 				top -= 1;
// 				strcpy(SS[top], buf);
// 				break;
// 				
// 			case MUL:
// 				strcpy(buf, "mul(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ",");
// 				strcat(buf, SS[top-1]);
// 				strcat(buf, ")");
// 				top -= 1;
// 				strcpy(SS[top], buf);
// 				break;
// 				
// 			case DIV:
// 				strcpy(buf, "div(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ",");
// 				strcat(buf, SS[top-1]);
// 				strcat(buf, ")");
// 				top -= 1;
// 				strcpy(SS[top], buf);
// 				break;
// 			case POW:
// 				strcpy(buf, "pow(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ",");
// 				strcat(buf, SS[top-1]);
// 				strcat(buf, ")");
// 				top -= 1;
// 				strcpy(SS[top], buf);
// 				break;	
// 				
// 			case LOG:
// 				strcpy(buf, "log(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ")");
// 				strcpy(SS[top], buf);
// 				break;
// 				
// 			case EXP:
// 				strcpy(buf, "exp(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ")");
// 				strcpy(SS[top], buf);
// 				break;
// 			case SIN:
// 				strcpy(buf, "sin(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ")");
// 				strcpy(SS[top], buf);
// 				break;
// 			case ASIN:
// 				strcpy(buf, "asin(");
// 				strcat(buf, SS[top]);
// 				strcat(buf, ")");
// 				strcpy(SS[top], buf);
// 				break;
// 			case NEG:
// 				strcpy(buf, "-");
// 				strcat(buf, SS[top]);
// 				strcat(buf, "");
// 				strcpy(SS[top], buf);
// 				break;	
// // 			case TAN:
// // 				strcpy(buf, "tan(");
// // 				strcat(buf, SS[top]);
// // 				strcat(buf, ")");
// // 				strcpy(SS[top], buf);
// // 				break;
// // 				
// 		}
// 	}
// 	
// 	printf("%s", SS[top]);
// }
// 
