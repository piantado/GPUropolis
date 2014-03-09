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




