






// __device__ op_t random_2arg_op(RNG_DEF) {
// 	return FIRST_2ARG + random_int(FIRST_1ARG-FIRST_2ARG, RNG_ARGS);
// }
// 
// __device__ op_t random_1arg_op(RNG_DEF) {
// 	return FIRST_1ARG + random_int(NUM_OPS-FIRST_1ARG, RNG_ARGS);
// }
// // returnt the *position* of a random 1-arg from program
// __device__ int sample_Narg_from(op_t* buf, int* posbuf, int nargs, int start, int end, RNG_DEF) {
// 	
// 	
// 	int idx = 0;
// 	for(int i=start;i<=end;i++){
// 		if(NARGS[buf[i]] == nargs){
// 			posbuf[idx] = i;
// 			idx++;
// 		}
// 	}
// 	
// 	if(idx==0) return -1;
// 	else return posbuf[random_int(idx, RNG_ARGS)]*(idx>0);
// }

// How many elements of x have an a-element argument?
// __device__ int count_Narg_in(op_t* x, int a, int start, int end) {
// 
// 	int cnt=0;
// 	for(int i=start;i<=end;i++){
// 		cnt += (NARGS[x[i]]==a);
// 	}
// 	return cnt;
// }

// // Set h2->program equal to h1->program
// __device__ void copy_program(hypothesis* h1, hypothesis* h2) {
// 	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
// 		h2->program[i] = h1->program[i];
// 	}
// }


// // shift everything [0...pos-1] one place to the right
// // This counts down so that we can use from=to
// __device__ void delete_at(op_t* from, op_t* to, int pos) {
// 	
// 	for(int i=MAX_PROGRAM_LENGTH-1;i>0;i--) {
// 		to[i] = from[i - (i<=pos)];
// 	}
// 	to[0] = NOOP;
// }

// // shift an array left, from pos, of length len. No guarantees on the value of the created middle bit
// __device__ void shift_array_left(op_t* from, op_t* to, int pos, int len) {
// 	
// 	// first shift everythign left:
// 	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
// 		to[i] = from[i+len*(i<pos)];
// 	}
// }
// 
// // at position pos, make a new space and insert (not respecting tree structure)
// __device__ void insert_at(op_t* from, op_t* to, int pos, op_t val) {
// 	
// 	shift_array_left(from,to,pos,1);
// 	to[pos] = val;
// }

// put ar endings (RHS) at positition pos, assuming ar has length len
// __device__ void insert_array_at(op_t* dest, int pos, op_t* ar, int len) {
// 	
// 	
// // 	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) {
// // 		
// // 	}
// 	
// 	for(int i=0;i<pos-len;i++) {
// 		dest[i] = dest[i+len];
// 	}
// 	int x=0;
// 	for(int i=pos-len;i<=pos;i++) {
// 		dest[i] = ar[x++];
// 	}
// }











/* =============================================================
 * Old-style (single-node) insert/delete proposals
 * ============================================================= 
 */

/*else if(pp==2) {
			
				// insert 1-arg
// 				int pos = (MAX_PROGRAM_LENGTH-1)-random_int(current->program_length-1, RNG_ARGS); // the position of what we replace; can't replace the first guy in the array
// 				
// 				op_t op = random_1arg_op(RNG_ARGS);
// 				
// 				insert_at(current->program, proposal->program, pos, op);
// 				
// 				compute_length_and_proposal_generation_lp(proposal);
// 				
// 				int start = MAX_PROGRAM_LENGTH-proposal->program_length;
// 				int end = MAX_PROGRAM_LENGTH-1;
// 				fb = nlog(current->program_length-1) + nlog((float)(NUM_OPS-FIRST_1ARG)) - nlog(count_Narg_in(proposal->program, 1,  start,end));
			}
			else if(pp == 3) {
				// Delete a 1-arg
			
				copy_program(current, proposal);
				
				int pos = sample_Narg_from(proposal->program, posbuf, 1, 0, MAX_PROGRAM_LENGTH-1, RNG_ARGS);
				
				if(idx >= 0) delete_at(proposal->program, proposal->program, pos);
				
				compute_length_and_proposal_generation_lp(proposal);
				int start = MAX_PROGRAM_LENGTH-proposal->program_length;
				int end = MAX_PROGRAM_LENGTH-1;
				
				fb = nlog(count_Narg_in(current->program, 1, start,end)) - (nlog(proposal->program_length) + nlog((float)(NUM_OPS-FIRST_1ARG)));
				*/
			}*/