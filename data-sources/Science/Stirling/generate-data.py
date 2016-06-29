# Generate data for trying to find Stirling's approximation. 
# SD is just fixed=1

import numpy
from math import sqrt

fac = 1
for i in xrange(1,10):
	fac = fac * i
	print i, float(fac), 1.0
	




