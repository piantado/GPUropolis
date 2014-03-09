/*
 * Here are some tips for speeding up CUDA: 	
 *   http://www.mindfiresolutions.com/Tips-for-speed-up-your-algorithm-in-the-CUDA-programming-399.php
 *  http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__MATH__INTRINSIC__SINGLE.html
 */

float f_output(float x, hypothesis* h, float* stack) {

	int top = 0;
	
	int program_start = MAX_PROGRAM_LENGTH-h->program_length;
	op_t* pi_ptr = h->program + program_start;
	
	// We could start at p=0 to avoid branching, but actually here we start at the program length since its faster for short programs
	for(int p=program_start;p<MAX_PROGRAM_LENGTH;p++) { // program pointer
		op_t op = (*pi_ptr);
		
		int newtop = top +  stack_change(op);
		
		// If top is out of range, this makes it a NOOP if we go out of range, preventing us from having to initialize the stack
		op = op*(top>=0)*(newtop>=0)*(top<MAX_PROGRAM_LENGTH)*(newtop<MAX_PROGRAM_LENGTH);
		
		switch(op){
// 			case NOOP: break;
// 			case ONE:  stack[newtop] = 1.0; break;
// 			case X:    stack[newtop] = x; break;
// 			case ADD:  stack[newtop] = __fadd_rz(stack[top], stack[top-1]); break; //stack[top] + stack[top-1]; break;
// 			case SUB:  stack[newtop] = __fadd_rz(stack[top], -stack[top-1]); break; //stack[top] - stack[top-1]; break;
// 			case MUL:  stack[newtop] = __fmul_rz(stack[top], stack[top-1]); break; // stack[top] * stack[top-1]; break;
// 			case DIV:  stack[newtop] = __fdiv_rz(stack[top], stack[top-1]); break; //stack[top] / stack[top-1]; break;
// 			case NEG:  stack[newtop] = -stack[top]; break;
// 			case POW:  stack[newtop] = __powf(stack[top], stack[top-1]);break;
// 			case LOG:  stack[newtop] = __logf(stack[top]);break;
// 			case EXP:  stack[newtop] = __expf(stack[top]);break;
// 			case SIN:  stack[newtop] = __sinf(stack[top]);break;
// 			case ASIN: stack[newtop] = asin(stack[top]);break;
// 			default: return 0.0/0.0; // if anything bad happens...
			case NOOP: break;
			case ONE:  stack[newtop] = 1.0; break;
			case X:    stack[newtop] = x; break;
			case ADD:  stack[newtop] = stack[top] + stack[top-1]; break;
			case SUB:  stack[newtop] = stack[top] - stack[top-1]; break;
			case MUL:  stack[newtop] = stack[top] * stack[top-1]; break;
			case DIV:  stack[newtop] = stack[top] / stack[top-1]; break;
			case NEG:  stack[newtop] = -stack[top]; break;
			case POW:  stack[newtop] = pow(stack[top], stack[top-1]);break;
			case LOG:  stack[newtop] = log(stack[top]);break;
			case EXP:  stack[newtop] = exp(stack[top]);break;
			case SIN:  stack[newtop] = sin(stack[top]);break;
			case ASIN: stack[newtop] = asin(stack[top]);break;
			default: return 0.0/0.0; // if anything bad happens...				
		}
		
		// update the top if we are not a NOOP, otherwise we run off the stack
		if(op > 0) top = newtop;
		
		pi_ptr++;
	}
	
	return stack[top];
}
