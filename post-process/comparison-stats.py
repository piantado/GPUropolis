"""plot.py

Usage:
	plot.py --directory=<DIR>  [--log]  [--trim=<L>]  [--traintype=<TT>]
  
Options:
 -h --help             Show this screen.
 --directory=<DIR>     The directory [default: run].
 --log                 Plot with log Y axis.
 --trim=<L>            Trim anything L log points from the MAP [default: 10].
 --traintype=<TT>      The training type: all, first-half, even-half [default: all].

"""

import matplotlib
import matplotlib.pyplot as plt
import os
import scipy.stats
import numpy

from Shared import *

from docopt import docopt
args = docopt(__doc__)
assert os.path.exists(args['--directory']), "Directory %s does not exist!"%args['directory']

NCURVEPTS = 50 # How many curve points to plot?
MIN_PLOT = 1e-4 # don't plot things lighter than this

## --------------------------------------
## Read the data file
## --------------------------------------

xs, ys, sds = load_data(args['--directory']+'/used-data/data.txt')

# TODO: Oh my god, we have to include the original x because it may matter for 
# fractional data points!! If we don't, then sometimes hypotehses fail to evaluate because
# the hypotheses have fractional powers that are not defined for fractional data.
# This was fixed most recently by disallowing negative fractional powers, but we may want to change that in the future...
newx = numpy.arange(0.99*min(xs), max(xs)*1.01, (1.01*max(xs)-0.99*min(xs))/float(NCURVEPTS))


## --------------------------------------
## Read the hypotheses
## --------------------------------------

H2cnt, H2post, H2f, H2firstfound = load_hypotheses(args['--directory']+'/samples.txt', trim=args['--trim'])

# sum up the posteriors
Z = logsumexp([ x[0] for x in H2post.values() ]) 
Zmax = max( [ x[0] for x in H2post.values() ])
Zcnt = sum(H2cnt.values())

# And compute the MAP
map_h, map_lp = None, float("-inf") # the map of the posteriors
for h in H2post.keys():
	if H2post[h] > map_lp: map_lp, map_h = H2post[h], h


print "# Loaded & compiled %i hypotheses!" % len(H2cnt.values())

## --------------------------------------
## Do fits and comparison, depending on the data type
## --------------------------------------

def ll(x,y,sd):
	return numpy.sum( scipy.stats.norm.logpdf( numpy.abs(x - y)/sd))

#indicator = all_data
LEN = len(xs)

traintype = args['traintype'] # How did we train?

for testtype in ['all', 'train', 'heldout']: # test on these
	
	
	# What data set do we train on?
	# NOTE: For our hypotheses, this is fixed by the GPU mcmc, NOT here
	if traintype=='all':         train_indicator = numpy.array( [True]*LEN )
	elif traintype=='first-half': train_indicator = numpy.arange(LEN) < (LEN/2)
	elif traintype=='even-half':  train_indicator = (numpy.arange(LEN) % 2) == 0
	else: assert False
	
	trainx   = xs[train_indicator]
	trainy   = ys[train_indicator]
	trainsds = sds[train_indicator]
	
	# What do we test on?		
	if testtype == 'all':      test_indicator = numpy.array( [True]*LEN )
	elif testtype=='train':    test_indicator = train_indicator
	elif testtype=='heldout':  test_indicator = ~train_indicator
	else: assert False
	
	testx   = xs[test_indicator]
	testy   = ys[test_indicator]
	testsds = sds[test_indicator]
	
	## Compute the polynomial fits:
	for degree in xrange(0,6):	
		coefs = numpy.polyfit(trainx, trainy, degree, w=1.0/(trainsds*trainsds)) ## inverse variance weighting polynomial fit
		
		p = numpy.poly1d(coefs)
		
		# And evaluate on the test
		print >>o, traintype, testtype, "P.%i.LL"%degree, ll( p(testx), testy, testsds)


	## AND PRINT OUT THE MODEL
	## TODO: RIGHT NOW THIS GIVES THE SAME ANSWER FOR ALL TRAINTYPE
	##       SINCE THAT IS DETERMINED IN THE GPU
	## The expected ll averaging via counts
	EcntLL = 0.0
	for h in H2cnt:	
		EcntLL += float(H2cnt[h]) * ll( H2f[h](testx), testy, testsds)
	EcntLL /= float(Zcnt)
	print >>o, traintype, testtype, "E.cnt.LL", EcntLL


	## The ll
	print traintype, testtype, "MAP.LL", ll(H2f[map_h](testx), testy, testsds)




