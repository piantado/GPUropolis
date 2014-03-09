# A cute script to run lz compression on coin flips
# and try to fit the resulting data

from LZ import encode

import random
import numpy

# generate random p values
ps = sorted([random.random() for i in xrange(20) ])

LEN = 100000 # how many coin flips -- must be long to overcome LZ's inefficiencies
REPS = 10 # how many times do we run each p?

for p in ps:
	
	reps = []
	for it in xrange(REPS):
		x = [str(1*(random.random()<p)) for i in xrange(LEN)]
		#print x
		#print LZ.encode(x,0)
		reps.append(  float(len(encode(x,0))) / float(LEN)  )
	print p, numpy.mean(reps), numpy.std(reps)