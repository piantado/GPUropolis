# Included fit function

def f(x):
	b1 = 1.9280693458E-01
	b2 = 1.9128232873E-01
	b3 = 1.2305650693E-01
	b4 = 1.3606233068E-01
	return b1*(x**2+x*b2) / (x**2+x*b3+b4)