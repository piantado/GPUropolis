/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Virtual machine for interpreting programs on the device
 */

__device__ data_t f_output(data_t X, hypothesis* h, data_t* stack) {

	int top = 0;
	
	int program_start = dMAX_PROGRAM_LENGTH-h->program_length;
	op_t* pi_ptr = h->program + program_start;
	
	int constant_i = 0; // index into the constant array, increasing order
	
	// We could start at p=0 to avoid branching, but actually here we start at the program length since its faster for short programs
	for(int p=program_start;p<dMAX_PROGRAM_LENGTH;p++) { // program pointer -- IF YOU CHANGE THIS, CHANGE IT IN THE PROGRAM PRINTER TO AVOID CONST WEIRDNESS
		op_t op = (*pi_ptr);
		
		int newtop = top + stack_change(op);
		
		// If top is out of range, this makes it a NOOP if we go out of range, preventing us from having to initialize the stack
		op = op*(top>=0 & newtop>=0 & top<dMAX_PROGRAM_LENGTH & newtop<dMAX_PROGRAM_LENGTH);
		
		switch(op){
			// Fast ops: http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__MATH__INTRINSIC__SINGLE.html
			case NOOP_: break;
			
			// This is generated by an external python system since C macros weren't quite up to the task
			#include "__VM_INCLUDE.cu"

			default: return CUDART_NAN_F; // if anything bad happens...
			
		}
		
		// update the top if we are not a NOOP, otherwise we run off the stack
		if(op > 0) top = newtop;
		
		// Hmm this shouldn't be needed, but seems to be?
		if(is_invalid(stack[top])) return CUDART_NAN_F;
		
		pi_ptr++;
	}
	
	return stack[top];
}
