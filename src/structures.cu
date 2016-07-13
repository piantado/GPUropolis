/*
  Structures and data types universally imported
*/


#ifndef SPEC_H
#define SPEC_H

typedef short op_t; // what type is the program primitives?
typedef float data_t; // what type is the data and stack?

// A struct for storing a hypothesis, which is a program and some accoutrements
// the check variables are there to make sure we don't overrun anything in any program editing
// main.cu has assertions to check they have CHECK_BIT value
// This also stores some ability for constants, but these are not used by all MH schemes
typedef struct hypothesis {
    float prior;
    float likelihood;
    float posterior;
    op_t   program[MAX_PROGRAM_LENGTH];
    int chain_index; // what index am I?
    float constants[MAX_CONSTANTS];
    int constant_types[MAX_CONSTANTS]; 
} hypothesis;

// A struct for storing data
typedef struct datum {
    data_t input;
    data_t output;
    data_t sd; // stdev of the output|input. 
} datum;

//  Specifications for an mcmc run. This way we can hand off different ones to different chains easily
typedef struct mcmc_specification {
    float acceptance_temperature;
    unsigned int iterations; // how many steps to run?
    int rng_seed;
    
    hypothesis proposal;
    data_t stack[STACK_SIZE];
    
    // and pointers to the data we want
    int data_length;
    datum* data;

    
} mcmc_specification;


typedef struct mcmc_results {
    
    hypothesis sample;
    hypothesis MAP;
    
    // for storing mcmc acceptance ratios, etc. 
    unsigned int acceptance_count;
    unsigned int proposal_count;
    
    int rng_seed; // what *was* I run with? Given by spec
    
} mcmc_results;

#endif