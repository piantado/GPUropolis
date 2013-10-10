"""

	TODO:
	 
"""

import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
from collections import defaultdict

import sympy
from sympy import init_printing, Symbol, expand
from sympy.parsing.sympy_parser import parse_expr
 
from Shared import *

parser = argparse.ArgumentParser(description='Plots for GPUropolis!')
parser.add_argument('--directory', dest='directory', type=str, default='runs/', nargs="?", help='The directory generated by GPUropolis')
parser.add_argument('--trim', dest='trim', type=float, default=100.0, nargs="?", help='Trim (completely ignore) anything this far from the MAP.')
parser.add_argument('--log', dest='log', type=str, default='', nargs="?", help='Should we make x and/or y a log scale?')
args = vars(parser.parse_args())

assert os.path.exists(args['directory']), "Directory %s does not exist!"%args['directory']

NCURVEPTS = 100 # How many curve points to plot?
MAX_CONSTANTS = 10 # only print out at most this many constants for each hypotheses
MIN_PLOT = 1e-5 # don't plot things lighter than this

FPregex = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?' # How to match floating points. Shoot me now.

## --------------------------------------
## Read the data file
## --------------------------------------

xs, ys, sds = [], [], []
for l in open(args['directory']+'/used-data/data.txt', 'r'):
	l = l.strip()
	if re.match("\s*#",l): continue
	
	x,y,sd = map(float, re.split("\s", l))
	
	xs.append(x)
	ys.append(y)
	sds.append(sd)
xs = numpy.array(xs)
ys = numpy.array(ys)
sds = numpy.array(sds)

if len(xs) == 0: newx = []
else:            newx = numpy.arange(min(xs), max(xs), (max(xs)-min(xs))/float(NCURVEPTS))


## --------------------------------------
## Read the hypotheses
## --------------------------------------


fullMaxLP = float("-inf")
for l in open(args['directory']+'/samples.txt', 'r'):
	if re.match("\s*#",l): continue
	parts = re.split("\t", l)
	lp = float(parts[2])
	if lp>fullMaxLP and not isnan(lp): fullMaxLP = lp
	

H2cnt = defaultdict(int)
H2post = dict()
H2f = dict()
H2firstfound = dict() # When did we first find this hypotheiss?
sn = -1 # what sample number are we on?
for l in open(args['directory']+'/samples.txt', 'r'):
	if re.match("\s*#",l): continue
	l = l.strip()
	parts = re.split("\t", l)
	
	rank, n, lp, prior, ll, h = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), parts[-1]
	lp = float(lp)
	
	# increment the sample number -- no longer is this reported explicitly
	if rank==0: sn += 1
	
	# Fully skip these -- not just in plotting. If they are too terrible
	if isnan(lp) or fullMaxLP-lp>args['trim']: continue
	
	h = re.sub("\"", "", h)
	
	if h not in H2cnt and not isnan(lp):
		
		## Store some things:
		H2cnt[h] += 1
		H2post[h] = map(float, [lp, prior, ll])
		H2firstfound[h] = sn
		
		try:
			H2f[h] = f = eval('lambda x: '+h)
		except MemoryError:
			H2f[h] = lambda x: float("nan")

# sum up the posteriors
Z = logsumexp([ x[0] for x in H2post.values() ]) 
Zmax = max( [ x[0] for x in H2post.values() ])
Zcnt = sum(H2cnt.values())

## --------------------------------------
## Now print out some stats:
## --------------------------------------

o = open(args['directory']+"/hypotheses.txt", 'w')
print >>o, "i first.found sample.count lp lpZ prior ll h hcon hform f0 f1 nconstants "+" ".join(['C%i'%i for i in xrange(MAX_CONSTANTS)])
for hi, h in enumerate( sorted(H2cnt.keys(), key=lambda x: H2post[x][0]) ):
	# And convert H to a more reasonable form:
	try:
		sympystr = str(parse_expr(h))
	except OverflowError: sympystr = "<OVERFLOW>"
	except ValueError:    sympystr = "<VALUE_ERROR>"
	except RuntimeError:  sympystr = "<RUNTIME_ERROR>"
	except MemoryError:   sympystr = "<MEMORY_ERROR>"
	
	# And store the constants
	constants = [ "NA" ] * MAX_CONSTANTS
	nconstants = 0
	for i,m in enumerate(re.findall(FPregex, sympystr)): 
		nconstants += 1
		if i < MAX_CONSTANTS: constants[i]=m # only store the first this many
	
	# Now print out a modified format for easier processing
	structural_form = re.sub(FPregex, 'C', sympystr) # The structural form, with no constants
	
	try: f0 = H2f[h](0.0)
	except: f0 = "NA"
	try: f1 = H2f[h](1.0)
	except: f1 = "NA"
	
	# Print out this hypothesis
	print >>o, hi, H2firstfound[h], H2cnt[h], H2post[h][0], H2post[h][0]-Z, H2post[h][1], H2post[h][2], q(h), q(sympystr), q(structural_form), f0,f1, nconstants, ' '.join(map(str,constants))	
o.close()

## --------------------------------------
## Make the plots
## --------------------------------------

for method in ['lpZ', 'maxratio', 'count']:

	fig = matplotlib.pyplot.figure(figsize=(5,4))
	plt = fig.add_subplot(1,1,1)
	
	for h in sorted(H2cnt.keys(), key=lambda x: H2post[x][0]):
		
		if method == 'count':      p = float(H2cnt[h]) / float(Zcnt)
		elif method == 'lpZ':      p = exp(H2post[h][0] - Z)
		elif method == 'maxratio': p = exp(H2post[h][0] - Zmax)
		print method, p, h
		
		if p < MIN_PLOT: continue
		
		newy = failmap(H2f[h], newx)
		
		assert any([ v is not None for v in newy]), "Failed to compute a valid value! Aborting! %s" % h
		
		try: 
			plt.plot(newx, newy, alpha=p, color="gray")
			#print newx, newy
			#print "Plotted!"
		except OverflowError: pass
		except ValueError: pass
		
	# Plot the data
	#print xs, ys
	plt.scatter(xs,ys, alpha=0.5, marker='.', color='red')
	plt.errorbar(xs, ys, yerr=sds, alpha=0.5, fmt=None, ecolor='red')	

	plt.set_xlim( *smartrange(xs) )
	
	plt.set_xlabel("Time")
	plt.set_ylabel("Distance")
	
	try:
		plt.set_ylim( *smartrange(ys, sds=sds)  )
	except OverflowError:
		print "** ERROR OVERFLOW IN YLIM:", max(ys), min(ys), ys, max(sds), min(sds), sds

	if re.search(r"x", args['log']): plt.set_xscale('log')
	if re.search(r"y", args['log']): plt.set_yscale('log')

	fig.savefig(args['directory']+"/plot-"+method+".pdf")

# Done

