/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * FUTURE: THIS IS NOT IMPLEMENTED/USED YET
 * Definitions of primitive operations and constants, and an ENUM to keep track of them all  
 */

// https://en.wikipedia.org/wiki/Physical_constant
// C_ = 2.99792458e8; // m s^-1
// G_ = 6.67384e-11; // M^3 kg^-1 s^-2
// h_ = 6.62606957e-34; // plank's constant

// Initialize the prior etc here:
const int MAX_NUM_OPS = 1000;

// arrays to store the prior on the host and device
float            hPRIOR[MAX_NUM_OPS]; 
__device__ float dPRIOR[MAX_NUM_OPS];

// lower two bits give the number of arguments
#define nargs(x) (x & 0x3);
#define stack_change(x) (1-(x) & 0x3)

// we use this to ensure we start at 0
#define COUNTER_START __COUNTER__

// make a macro to define this like an enum, but with the lower two bits holding the number of arguments
#define DEFINE_OP(name, narg) const op_t name = ( (__COUNTER__-COUNTER_START) << 2) | narg;
	
/* So that if we call
   DEFINE_OP(EXP_, 2, exp) 
   we will create:
	EXP_ -> a unique integer whose lower two bits are the number of arguments
	strEXP_ -> a string of "EXP_" for matchign later
*/

NAME NARGS RHS
// should be zero
DEFINE_OP( NOOP_, 0)
#define fNOOP_ break;

// constants should come first here since they will tend to be higher probability
DEFINE_OP( X_, 0)
#define f

DEFINE_OP( ONE_, 0)

DEFINE_OP( NEG_, 1)
DEFINE_OP( LOG_, 1)
DEFINE_OP( EXP_, 1)
DEFINE_OP( SIN_, 1)
DEFINE_OP( ASIN_, 1)

DEFINE_OP( ADD_, 2)  
DEFINE_OP( SUB_, 2)
DEFINE_OP( MUL_, 2)
DEFINE_OP( DIV_, 2)
DEFINE_OP( POW_, 2)

__device__ int NUM_OPS = __COUNTER__;

int* USED_PRIMITIVES[] = {NOOP_, ONE_, NEG_, LOG_, EXP_, SIN_, ASIN_, ADD_, SUB_, MUL_, DIV_, POW_};
char* OP_NAMES[]       = {"NOOP_", "ONE_", "NEG_", "LOG_", "EXP_", "SIN_", "ASIN_", "ADD_", "SUB_", "MUL_", "DIV_", "POW_"};



