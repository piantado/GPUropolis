import numpy

from math import log, exp, sin, pow

def generate_data(name, f, xs, sd=0.0):
	# Put into the file the f of each xs, with error noise sd
	
	err = numpy.zeros(len(xs)) if sd == 0.0 else numpy.random.normal(0,sd,len(xs))
	ys = numpy.array(map(f, xs)) + err
	
	o = open(name, 'w')
	for x,y in zip(xs, ys):
		print >>o, x, "\t", y
	o.close()
	


xs = numpy.arange(1,25, 1)

generate_data("datasets/inverse-U.txt", lambda x: -(x-15)**2+1029.0, xs, 1.0)

generate_data("datasets/gaussian.txt", lambda x: exp( -(x-10.)**2.0 ), xs, 0.0)

# a nice scatter of sin curves, which are hard to interpret on the x locations
generate_data("datasets/sin-scatter.txt", lambda x: sin(x), xs, 0.0)

generate_data("datasets/sinlog.txt", lambda x: sin(log(x)), xs, 0.0)

generate_data("datasets/sinlog1x.txt", lambda x: sin(log(x/(1.0+x))), xs, 0.0)

generate_data("datasets/linear-neg.txt", lambda x: 312. - 14.*x, xs, 0.0)

generate_data("datasets/sinx25powx.txt", lambda x: sin((x/25.)**x), xs, 0.0)

generate_data("datasets/logit.txt", lambda x: log( x / (30.-x)), xs, 0.0)

generate_data("datasets/logpsin.txt", lambda x: log( x + exp(1./x)) + sin(14./x), xs, 0.0)





