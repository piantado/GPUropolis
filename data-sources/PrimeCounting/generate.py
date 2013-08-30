
from math import log

"""
	Stackoverflow: http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python
"""
def get_primes(n):
	numbers = set(range(n, 1, -1))
	primes = []
	while numbers:
		p = numbers.pop()
		primes.append(p)
		numbers.difference_update(set(range(p*2, n+1, p)))
	return primes


N = 100000
primes = { i:True for i in get_primes(N) }
for i in xrange(10000,N+1,2000):
	cnt = len([x for x in primes if x<=i ])
	print i, "\t", cnt, "\t", cnt*(1+1.0/log(i)+2.51/(log(x)**2.)) - cnt # using Pierre Dusart's simple bound as a stdev