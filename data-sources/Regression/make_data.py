import numpy
from numpy.random import random, normal
import os
import sys
from math import pi, tan, radians, cos
import itertools
import datetime

# Make simple examples of linear regression with varying slopes, to estimate bias and distribution 
# of estimator as a function of slope (constant value) and number of data points

N = 10

# Parameterize the slope in terms of angles
ANGLES = range(0, 85+1, 5) 
Ns     = [25,100]
SD = 0.1

print ANGLES
for n, theta in itertools.product(Ns,ANGLES):
	x = random(n)*cos(radians(theta)) # uniform on [0,sin(theta)] (to keep x,y in the same range)
	y = x * tan(radians(theta)) + normal(0.0, SD, size=n) # add normal noise with this sd
	
	d = "%i_%i" %(theta,n)
	os.mkdir(d)
	
	f = open(d+"/data.txt", 'w')
	print >>f, "# Created on: " + datetime.datetime.now().isoformat(' ')
	print >>f, "# degrees: %i" % theta
	print >>f, "# data_points: %i" % n
	print >>f, "# sd: %f" % SD
	
	for i in xrange(n):
		print >>f, "%f\t%f\t%f" % (x[i], y[i], SD)
	f.close()
