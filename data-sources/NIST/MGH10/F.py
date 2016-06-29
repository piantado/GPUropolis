# Included fit function

from math import exp

def f(x):
	b1 = 5.6096364710E-03
	b2 = 6.1813463463E+03
	b3 = 3.4522363462E+02 
	return b1 * exp(b2/(x+b3))