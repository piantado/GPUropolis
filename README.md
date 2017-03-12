 
GPUropolis
==========

GPUropolis is software for using CUDA to run MCMC on compositional spaces. The current version implements a lightweight expression evaluator in CUDA and runs Metropolis-hastings on both the form of arithmetic expresssions (equations) and their constants (parameters) as a CUDA kernel. Defaulty, it uses a description lenght prior over expressions, a cauchy prior on constant values, Gaussian proposals, and resamples chains that fall too far below the current sample MAP. 

GPUropolis is under heavy development the computation and language lab. 

Distributed under GPL3.

As of March 2017, a new update has been released that simplifies code and coalesces memory access for much faster sampling times. 

Input/output:
------------
A standard run produces:
* samples.txt - every chain's current state at the end of a block
* log.txt - logging information, including the rng seed and parameters
		
