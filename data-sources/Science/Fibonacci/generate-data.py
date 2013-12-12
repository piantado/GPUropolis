

N = 25
 
fib = [None]*N
fib[0] = 1
fib[1] = 1

for i in xrange(2,N):
	fib[i] = fib[i-1]+fib[i-2]
	
for i in xrange(N):
	print i, "\t", fib[i], "\t", 0.01