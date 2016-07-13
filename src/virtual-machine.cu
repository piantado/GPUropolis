/*
 * Stack-based virtual machine for interpreting programs on the device
 * 
 * TODO: Re-write all stack stuff with pointers?
 */
#include "structures.cu"

const int NUM_OPS=16;
enum OPS { NOOP_=0,CONSTANT_=1,X_=2,ZERO_=3,ONE_=4,EXP_=5,LOG_=6,NEG_=7,MUL_=8,POW_=9,ADD_=10,DIV_=11,SUB_=12, RET_=13, DUP_=14, SWAP_=15};

__device__ data_t f_output(data_t X, hypothesis* h, data_t* stack) {

    // zero the stack
    for(int s=0;s<STACK_SIZE;s++) {
        stack[s] = 0;
    }
    
    // used if we have constants
    int constant_i = 0; // index into the constant array, increasing order
    int top = STACK_START; // move to where we start
    data_t tmp;
    
    op_t* pi_ptr = h->program;
    for(int p=0;p<MAX_PROGRAM_LENGTH;p++){
        op_t op = (*pi_ptr);
        
        switch(op){
            
            // these push onto the stack
            case CONSTANT_: 
                top++; 
                stack[top] = h->constants[constant_i%MAX_CONSTANTS]; 
                constant_i++;
                break;
            case X_:    top++; stack[top] = X; break;
            case ZERO_: top++; stack[top] = 0.0f; break;
            case ONE_:  top++; stack[top] = 1.0f; break;
            
            // These leave the stack unchanged
            case EXP_: stack[top] = expf(stack[top]); break;
            case LOG_: stack[top] = logf(stack[top]); break;
            case NEG_: stack[top] = -(stack[top]); break;
            
            // these pop one off the stack
            case MUL_: top--; stack[top] = (stack[top+1] * stack[top]); break;
            case POW_: top--; stack[top] = powf(stack[top+1],stack[top]); break;
            case ADD_: top--; stack[top] = (stack[top+1] + stack[top]); break;
            case DIV_: top--; stack[top] = (stack[top+1] / stack[top]); break;
            case SUB_: top--; stack[top] = (stack[top+1] - stack[top]); break;
            
            // control flow
            case RET_:  goto DONE; 
            case DUP_:  top++; stack[top] = stack[top-1]; break;
            case SWAP_: 
                tmp = stack[top];
                stack[top] = stack[top-1];
                stack[top-1] = tmp;
                break;
            case NOOP_: break; // do nothing
            
            default: return CUDART_NAN_F; // if anything bad happens...
        }
        
        pi_ptr++;
    }
    
DONE:
    
//     cuPrintf("HERE %i\n", stack[top]);
    
// 	if(is_invalid(stack[top])) return CUDART_NAN_F; // Hmm seems necessary?
	
    return stack[top];
}
