# Hmm what do we do about the residual variance. Should it scale with the magnitude (e.g. a proportion variance?)

import numpy
from numpy.random import random, normal
import os
import sys
from math import pi, tan, radians
import itertools
import datetime

# Simple estimator of constants -- check out bias and variance


SD = 0.1
Ns = [10, 20, 100]
Vs = [0, 1, 2, 10, 20, 100, 200, 1000, 2000]

for v, n in itertools.product(Vs,Ns):
	
	y = normal(v, SD, size=n) # add normal noise with this sd
	
	d = "%i_%i" % (v,n)
	os.mkdir(d)
	
	f = open(d+"/data.txt", 'w')
	print >>f, "# Created on: " + datetime.datetime.now().isoformat(' ')
	print >>f, "# v: %i" % v
	print >>f, "# n: %i" % n
	print >>f, "# sd: %f" % SD
	
	for i in xrange(n):
		print >>f, "%f\t%f\t%f" % (0.0, y[i], SD)
	f.close()
