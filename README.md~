 
GPUropolis
==========

GPUropolis is software for using CUDA to run MCMC on compositional spaces. At present it runs symbolic regression on a variety of historical data sets. 

GPUropolis is under heavy development by colala, the computation and language lab at the University of Rochester. Constact Steve Piantadosi (spiantadosi@bcs.rochester.edu) for questions. 

Distributed under GPL3.

Input/output:
------------
A standard run produces several files:
* samples.txt - every chain's current state at the end of a block
* MAPs.txt - every chain's best found hypothesis at the end of a block 
* tops.txt - the Top NTOP (default 5k) hypotheses found overall
* log.txt - logging information, including the rng seed and parameters
* performance.txt - performance statistics -- samples/second, function evals/second, etc. 
		
Current sampling schemes:
------------------------
	
MH_kernel stats at a temperature given by the maximum likelihood found so far (using the max possible for iteration 0). It anneals down to 1.0 using half of the MCMC_ITERATIONS and runs at 1.0 for the second half. This is called one "block" of samples. The sampler will run for OUTER_BLOCKS number of blocks, with BURN_BLOCKS before (data is not saved from BURN_BLOCKS). 

At the end of each block, the sampler can do one of several things (as given by END_OF_BLOCK_ACTION): 
* 1: start anew each outer loop (restart from prior)
* 2: maintain the same chain (default)
* 3: resample via current probability at a new temperature via RESAMPLE_PRIOR_TEMPERATURE/ RESAMPLE_LIKELIHOOD_TEMPERATURE
* 4: resample from the global top hypotheses (also using  RESAMPLE_PRIOR_TEMPERATURE/ RESAMPLE_LIKELIHOOD_TEMPERATURE)

There is also an experimental search kernel, and a simple one to sample from the prior. 

Languages:
----------

There are several possible languages specifying primitives and functions. All correspond to grammars with only one nonterminal type (EXPR). The languages can be changed via make_primitives_header.py, which outputs src/__PRIMITIVES.cu and src/__VM_INCLUDE.cu, giving the primitives and prior, and virtual machine instructions, respectively. make_primitives_header.py takes command line arguments of any of the files in languages. languages/Polynomial.txt should always be included. 

These files have the format of columns for: name, number of arguments (0 for constants), stack operation. The stack operation is substituted into __VM_INCLUDE.cu and is C code that gives what should be put on top of the stack (see __VM_INCLUDE.cu for examples of how these are translated).