 
GPUropolis
==========

GPUropolis is software for using CUDA to run MCMC on compositional spaces. The current version implements a lightweight stack-based virtual machine in CUDA and runs a version of Metropolis-hastings on both the form of arithmetic expresssions (equations) and their constants as a CUDA kernel.  

GPUropolis is under heavy development the computation and language lab at the University of Rochester. 

Distributed under GPL3.

Input/output:
------------
A standard run produces several files:
* samples.txt - every chain's current state at the end of a block
* MAPs.txt - every chain's best found hypothesis at the end of a block 
* log.txt - logging information, including the rng seed and parameters
* performance.txt - performance statistics -- samples/second, function evals/second, etc. 
		
