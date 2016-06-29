# Included fit function

from math import exp

def f(x):
	b1 = 6.9964151270E+02
	b2 = 5.2771253025E+00
	b3 = 7.5962938329E-01
	b4 = 1.2792483859E+00
	return b1 / ((1+exp(b2-b3*x))**(1/b4))