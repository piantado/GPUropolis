# Included fit function

from math import exp

def f(x):
	b1 = 1.5543827178E+00
	b2 = 4.0888321754E+00
	b3 = 4.5154121844E+02
	return (b1/b2) * exp(-0.5*((x-b3)/b2)**2)