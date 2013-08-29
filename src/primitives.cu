/*

	Definitions of primitive operations and constants, and an ENUM to keep track of them all
 
*/

// https://en.wikipedia.org/wiki/Physical_constant
// C_ = 2.99792458e8; // m s^-1
// G_ = 6.67384e-11; // M^3 kg^-1 s^-2
// h_ = 6.62606957e-34; // plank's constant
// 

// Number theory: mod,etc, maybe have it work on ints
// Physics: 
// Mathematics

// char* ALL_OPS[100];
// OP[0]="ADD"; 
// OP[0]="ADD"; 


int idx = 0x0;

// make a macro to define this like an enum, but with the lower two bits holding the number of arguments
#define DEFOP(name,narg) const op_t name = (idx << 2) | narg; ALL_OPS[idx] = "name"; idx++;

DEFOP( X, 0)
DEFOP( 1, 0)

DEFOP( NEG, 1)
DEFOP( LOG, 1)
DEFOP( EXP, 1)
DEFOP( SIN, 1)
DEFOP( ASIN, 1)

DEFOP( ADD, 2 )  
DEFOP( SUB, 2 )
DEFOP( MUL, 2 )
DEFOP( DIV, 2 )
DEFOP( POW, 2 )




