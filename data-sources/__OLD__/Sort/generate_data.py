
# A cute example -- run sort multiple times
# And see if you can discover the time complexity of python sort

import os
import time
import numpy
import random

i = 0
Ns = xrange(1,100) # TIMES a million below!
REPS = 10

for n in Ns:
	
	mytimes = []
	
	
	for r in xrange(REPS):
		
		
		x = [random.random() for i in xrange(n*1000000) ]
		start = time.clock()
		v = sorted(x)	
		mytimes.append( (time.clock() - start) )
	
	print n, numpy.mean(mytimes), numpy.std(mytimes)
	