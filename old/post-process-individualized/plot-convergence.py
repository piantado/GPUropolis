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

NCURVEPTS = 100 # How many curve points to plot?
MIN_PLOT = 1e-4 # don't plot things lighter than this

## --------------------------------------
## Read the hypotheses -- we can do this pretty simply, not using the library function
## --------------------------------------

posteriors = []

this_posteriors = []
for l in open(args['--directory']+'/samples.txt', 'r'):
	if re.match("\s*#",l): continue
	l = l.strip()
	parts = re.split("\t", l)
	
	rank, n, lp, prior, ll, h = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), parts[-1]
	lp = float(lp)
	
	
	if rank == 0 and len(this_posteriors)>0:
		posteriors.append(numpy.array(this_posteriors))
		this_posteriors = []
		
	this_posteriors.append(lp)

# Now plot


TODO: FINISH THIS