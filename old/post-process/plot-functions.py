"""plot.py

Usage:
	plot.py --directory=<DIR>  [--log=<axes>]  [--trim=<L>]
  
Options:
 -h --help             Show this screen.
 --directory=<DIR>     The directory [default: run].
 --log=<axes>          Plot with log axes [default: ''].
 --trim=<L>            Trim anything L log points from the MAP [default: 10].

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

NCURVEPTS = 100 # How many curve points to plot?
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
## Make the plots
## --------------------------------------

if not os.path.exists(args['--directory']+"/plots/"): os.mkdir( args['--directory']+"/plots/" )

for method in ['lpZ', 'maxratio', 'count', 'one']:

	fig = matplotlib.pyplot.figure(figsize=(5,4))
	plt = fig.add_subplot(1,1,1)
	
	NH = len(H2cnt.keys())
	
	for h in sorted(H2cnt.keys(), key=lambda x: H2post[x][0]):
		
		if method == 'count':      p = log(H2cnt[h]+1) / log(Zcnt + NH)  # color on log scale
		elif method == 'one':      p = min(1.0, 10.0 / float(NH) )
		elif method == 'lpZ':      p = exp(H2post[h][0] - Z)
		elif method == 'maxratio': p = exp(H2post[h][0] - Zmax)
		
		if p < MIN_PLOT: continue
		print method, p, h
		
		plotx, ploty = failmap(H2f[h], newx)
		
		## Hmm check here that we can get some values!
		if len(ploty)==0:
			print "Failed to compute any valid value! Aborting! %s" % h
			continue
		
		try: 
			plt.plot(plotx, ploty, alpha=p, color="gray", zorder=1)
			#print "Plotted!"
		except OverflowError: pass
		except ValueError: pass
		

	## NOTE: These have to come before the data or sometimes it gets angry
	plt.set_xlim( *smartrange(xs) )
	plt.set_ylim( *smartrange(ys, sds=sds)  )
	
	# Plot the data
	plt.scatter(xs ,ys, alpha=1.0, marker='.', color='red', zorder=10)
	plt.errorbar(xs, ys, yerr=sds, alpha=1.0, fmt=None, ecolor='red', zorder=9)	

	plt.set_xlabel("Time")
	plt.set_ylabel("Distance")
	
	
	if re.search(r"x", args['--log']): plt.set_xscale('log')
	if re.search(r"y", args['--log']): plt.set_yscale('log')

	fig.savefig(args['--directory']+"/plots/"+method+".pdf")

print "# Done plotting."




