"""

	TODO:
	 
"""

import matplotlib
import matplotlib.pyplot as plt
import argparse

from collections import defaultdict

import sympy
from sympy import init_printing, Symbol, expand
from sympy.parsing.sympy_parser import parse_expr

from Shared import *

parser = argparse.ArgumentParser(description='MCMC for Baboon data!')
parser.add_argument('--in', dest='in', type=str, default='o.txt', nargs="?", help='The input sample file')
parser.add_argument('--data', dest='data', type=str, default='data.txt', nargs="?", help='The input data file')
parser.add_argument('--out', dest='out', type=str, default='run', nargs="?", help='The output directory')
parser.add_argument('--log', dest='log', type=str, default='', nargs="?", help='Should we make x and/or y a log scale?')
args = vars(parser.parse_args())

NCURVEPTS = 250 # How many curve points to plot?
MAX_CONSTANTS = 10 # only print out at most this many constants for each hypotheses
MIN_PLOT = 1e-5 # don't plot things lighter than this
FULL_TRIM_BELOW = 500; # more than this many log points below the best are NOT even loaded (or processed)
FPregex = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?' # How to match floating points

## --------------------------------------
## Read the data file
## --------------------------------------

xs, ys, sds = [], [], []
for l in open(args['data'], 'r'):
	l = l.strip()
	if re.match("\s*#",l): continue
	
	x,y,sd = map(float, re.split("\t", l))
	
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
for l in open(args['in'], 'r'):
	if re.match("\s*#",l): continue
	parts = re.split("\t", l)
	lp = float(parts[4])
	if lp>fullMaxLP: fullMaxLP = lp

H2cnt = defaultdict(int)
H2post = dict()
H2f = dict()
H2firstfound = dict() # When did we first find this hypotheiss?
for l in open(args['in'], 'r'):
	if re.match("\s*#",l): continue
	l = l.strip()
	parts = re.split("\t", l)
	
	ismap, sn, n, rank, lp, prior, ll, h = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[-1]
	lp = float(lp)
	
	# Fully skip these -- not just in plotting. If they are too terrible
	if isnan(lp) or fullMaxLP-lp>FULL_TRIM_BELOW: continue
	
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

o = open(args['out']+"/hypotheses.txt", 'w')
print >>o, "i first.found sample.count lp lpZ prior ll h hcon hform f0 f1 nconstants "+" ".join(['C%i'%i for i in xrange(MAX_CONSTANTS)])
for hi, h in enumerate(H2cnt):
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

	for h in H2cnt:
		
		if method == 'count': p = float(H2cnt[h]) / float(Zcnt)
		elif method == 'lpZ': p = exp(H2post[h][0]-Z)
		elif method == 'maxratio': p = exp(H2post[h][0] - Zmax)
		#print method, p, h
		
		if p < MIN_PLOT: continue
		
		newy = failmap(H2f[h], newx)
		
		try: 
			plt.plot(newx, newy, alpha=p, color="gray")
		except OverflowError: pass
		except ValueError: pass
		
	# Plot the data
	plt.scatter(xs,ys, alpha=0.5, marker='.', color='red')
	plt.errorbar(xs, ys, yerr=sds, alpha=0.5, fmt=None, ecolor='red')	

	plt.set_xlim( *smartrange(xs) )
	
	try:
		plt.set_ylim( *smartrange(ys, sds=sds)  )
	except OverflowError:
		print "** ERROR OVERFLOW IN YLIM:", max(ys), min(ys), ys, max(sds), min(sds), sds

	if re.search(r"x", args['log']): plt.set_xscale('log')
	if re.search(r"y", args['log']): plt.set_yscale('log')

	fig.savefig(args['out']+"/plot-"+method+".pdf")

# Done

