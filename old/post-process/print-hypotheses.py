"""plot.py

Usage:
	plot.py --directory=<DIR>  [--log]  [--trim=<L>] 
	
Options:
 -h --help             Show this screen.
 --directory=<DIR>     The directory [default: run].
 --log                 Plot with log Y axis.
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

## Read the hypotheses
H2cnt, H2post, H2f, H2firstfound = load_hypotheses(args['--directory']+'/samples.txt', trim=args['--trim'])
print "# Loaded & compiled %i hypotheses!" % len(H2cnt.values())

Z = logsumexp([ x[0] for x in H2post.values() ]) 

# --------------------------------------
# Now print out some stats:
# --------------------------------------

Hform2constants = defaultdict(list) # a list of lists of constants for each structural form
Hform2best = dict() # The best hypothesis for each structural form

print "i first.found sample.count lp lpZ prior ll h hcon hform f0 f1 nconstants "+" ".join(['C%i'%i for i in xrange(MAX_CONSTANTS)])
for hi, h in enumerate( sorted(H2cnt.keys(), key=lambda x: H2post[x][0], reverse=True) ):
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
		if i < MAX_CONSTANTS: constants[i]=float(m) # only store the first this many
	
	# Now print out a modified format for easier processing
	structural_form = re.sub(FPregex, 'C', sympystr) # The structural form, with no constants
	
	# Store these as constants for this form
	Hform2constants[structural_form].append(constants)
	
	# Store the best value for this form
	if structural_form not in Hform2best or H2post[h][0] > Hform2best[structural_form]:
		Hform2best[structural_form] = H2post[h][0]
	
	try: f0 = H2f[h](0.0)
	except: f0 = "NA"
	try: f1 = H2f[h](1.0)
	except: f1 = "NA"
	
	# Print out this hypothesis
	#print hi, H2firstfound[h], H2cnt[h], H2post[h][0], H2post[h][0]-Z, H2post[h][1], H2post[h][2], q(h), q(sympystr), q(structural_form), f0,f1, nconstants, ' '.join(map(str,constants))	


# --------------------------------------
# Make histograms of each constant for each ranked hypothesis
# Giving the distribution of that constant, given the rank of the structural form
# --------------------------------------

constantplotpath = args['--directory']+"/constants/"
if not os.path.exists(constantplotpath): os.mkdir(constantplotpath)

for r, form in enumerate(sorted(Hform2best.keys(), key=lambda x: Hform2best[x], reverse=True)):
	
	# Make for the rth rank
	if not os.path.exists(constantplotpath+"/%i/"%r): os.mkdir(constantplotpath+"/%i/"%r )
	
	# Now plot each constant
	for ci in range(MAX_CONSTANTS):
		if Hform2constants[form][0][ci] is "NA": break # don't do anything
		
		fig = matplotlib.pyplot.figure(figsize=(4,4))
		plt = fig.add_subplot(1,1,1)
		
		hist, bins = numpy.histogram(map(lambda x: x[ci], Hform2constants[form]), bins=50)
		width = 0.7 * (bins[1] - bins[0])
		center = (bins[:-1] + bins[1:]) / 2
		plt.bar(center, hist, align='center', width=width)
		
		fig.savefig(constantplotpath+"/%i/%i.pdf"%(r,ci))
			
			
	# Only analyze this far
	if r > 10: break 