# Generate data for the binomial -> Gaussian

import numpy
from math import sqrt

K = 10 # the max of the binomial
N = 10000 # How many samples to generate

bins = numpy.bincount(numpy.random.binomial(K,0.5, size=N), minlength=K)

for i,cnt in enumerate(bins):
	pi = float(cnt)/float(N)
	print i, cnt, sqrt( float(N)*pi*(1.-pi) )




