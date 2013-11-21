/* GENERATED BY make_primitives_header.py in Makefile. DO NOT EDIT */


const int NUM_OPS=16;

enum OPS { NOOP_=0,X_=1,ONE_=2,SQRT_=3,NEG_=4,SGN_=5,GAMMA_=6,ABS_=7,ROUND_=8,LOG_=9,EXP_=10,SUB_=11,DIV_=12,MUL_=13,ADD_=14,POW_=15};

__device__ const int NARGS[]  = {0, 0,0,1,1,1,1,1,1,1,1,2,2,2,2,2 };
 const int hNARGS[] = {0, 0,0,1,1,1,1,1,1,1,1,2,2,2,2,2 };
 const char* NAMES[] = { "<NA>", "x","one","sqrt","neg","sgn","gamma","abs","round","log","exp","sub","div","mul","add","pow" };

 // Non-defined primitives, used potentially by print_program_as_expression

#define CONSTANT_ -99
