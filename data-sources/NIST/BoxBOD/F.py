# Included fit function

from math import exp

def f(x):
	b1 = 2.1380940889E+02
	b2 = 5.4723748542E-01
	return b1*(1-exp(-b2*x))