"""postprocess.py

Usage:
	postprocess.py --directory=<DIR>  [--log=<axes>]  [--trim=<L>] [--train-type=<TT>]
  
Options:
 -h --help             Show this screen.
 --directory=<DIR>     The directory [default: run].
 --log=<axes>          Plot with log axes [default: ''].
 --trim=<L>            Trim anything L log points from the MAP [default: 10].
 --train-type=<TT>      The training type: all, first-half, even-half [default: all].
"""

"""

	A new version to do all postprocessing at the same time.
	
	TODO: 
		- Plot Held-out points differently (Filled vs open)
		- Make this extract trainigntype from the directory string
		- only plot the heldout analysis! But put training accuracy in there too!
		- Fix the lieklihood to respect noisy predictions
		- A plot of observed-vs-expected probabilities for hypotheses -- how many in each bin?
		- Check our estimator from counts 
		- Need an "estimator" type to be accepted as input
		- Fix pow that plots offscreen -- handle inf in error curve plots
"""
import os
import sys
import numpy
import scipy
import scipy.stats
from sklearn.gaussian_process import GaussianProcess
import matplotlib
import matplotlib.pyplot as pyplot

from Shared import *

from docopt import docopt
args = docopt(__doc__)
assert os.path.exists(args['--directory']), "Directory %s does not exist!"%args['--directory']


DIRECTORY = args['--directory']
DATA_FILE = DIRECTORY+'/used-data/data.txt'
HYPOTHESIS_FILE = DIRECTORY + "/samples.txt"
PLOT_DIR = DIRECTORY+"/plots/"

trim = float(args['--trim'])
traintype = args['--train-type'] # How did we train?

NCURVEPTS = 100 # How many curve points to plot?
MIN_PLOT = 1e-4 # don't plot things lighter than this


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Loads a data file, setting the x,y,sd triple. 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

xs, ys, sds = [], [], []
for l in open(DATA_FILE, 'r'):
	l = l.strip()
	if re.match("\s*#",l): continue
	
	x,y,sd = map(float, re.split("\s+", l))
	
	xs.append(x)
	ys.append(y)
	sds.append(sd)
xs = numpy.array(xs)
ys = numpy.array(ys)
sds = numpy.array(sds)

# TODO: Oh my god, we have to include the original x because it may matter for 
# fractional data points!! If we don't, then sometimes hypotehses fail to evaluate because
# the hypotheses have fractional powers that are not defined for fractional data.
# This was fixed most recently by disallowing negative fractional powers, but we may want to change that in the future...
newx = numpy.arange(0.99*min(xs), max(xs)*1.01, (1.01*max(xs)-0.99*min(xs))/float(NCURVEPTS))
LEN = len(xs)

#"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#Load the hypotheses
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#"""

fullMaxLP = float("-inf")
for l in open(HYPOTHESIS_FILE, 'r'):
	if re.match("\s*#",l): continue
	parts = re.split("\t", l)
	lp = float(parts[2])
	if lp>fullMaxLP and not isnan(lp): fullMaxLP = lp
	
print "# Max posterior found:", fullMaxLP

H2cnt = defaultdict(int)
H2post = dict()
H2f = dict()
H2firstfound = dict() # When did we first find this hypotheiss?
map_h, map_lp = None, float("-inf") # the map of the posteriors

sn = -1 # what sample number are we on?
for l in open(HYPOTHESIS_FILE, 'r'):
	if re.match("\s*#",l): continue
	l = l.strip()
	parts = re.split("\t", l)
	
	rank, n, lp, prior, ll, h = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), parts[-1]
	
	# increment the sample number -- no longer is this reported explicitly
	if rank==0: sn += 1
	
	# Fully skip these -- not just in plotting. If they are too terrible
	if isnan(lp) or fullMaxLP-lp>trim: continue
	h = re.sub("\"", "", h)
	
	if h not in H2cnt and not isnan(lp):
		
		H2cnt[h] += 1
		H2post[h] = map(float, [lp, prior, ll])
		H2firstfound[h] = sn
		
		# Keep track of the MAP hypothesis
		if lp > map_lp: map_lp, map_h = lp, h
		
		try:
			H2f[h] = f = eval('lambda x: '+h)
		except MemoryError:
			H2f[h] = lambda x: float("nan")

print "# Loaded & compiled %i hypotheses!" % len(H2cnt.values())

# sum up the posteriors
Z = logsumexp([ x[0] for x in H2post.values() ]) 
Zmax = max( [ x[0] for x in H2post.values() ])
Zcnt = sum(H2cnt.values())




#"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#Plot the curves
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#"""

#if not os.path.exists(PLOT_DIR): os.mkdir( PLOT_DIR )

#for method in ['lpZ', 'maxratio', 'count', 'one']:

	#fig = matplotlib.pyplot.figure(figsize=(5,4))
	#plt = fig.add_subplot(1,1,1)
	
	#NH = len(H2cnt.keys())
	
	#for h in sorted(H2cnt.keys(), key=lambda x: H2post[x][0]):
		
		#if method == 'count':      p = log(H2cnt[h]+1) / log(Zcnt + NH)  # color on log scale
		#elif method == 'one':      p = min(1.0, 10.0 / float(NH) )
		#elif method == 'lpZ':      p = exp(H2post[h][0] - Z)
		#elif method == 'maxratio': p = exp(H2post[h][0] - Zmax)
		
		#if p < MIN_PLOT and method in set(['lpZ', 'maxratio']): continue
		#print method, p, h
		
		#plotx, ploty = failmap(H2f[h], newx)
		
		### Hmm check here that we can get some values!
		#if len(ploty)==0:
			#print "Failed to compute any valid value! Aborting! %s" % h
			#continue
		
		#try: 
			#plt.plot(plotx, ploty, alpha=p, color="gray", zorder=1)
			##print "Plotted!"
		#except OverflowError: pass
		#except ValueError: pass
		

	### NOTE: These have to come before the data or sometimes it gets angry
	#plt.set_xlim( *smartrange(xs) )
	#plt.set_ylim( *smartrange(ys, sds=sds)  )
	
	## Plot the data
	#plt.scatter(xs ,ys, alpha=1.0, marker='.', color='red', zorder=10)
	#plt.errorbar(xs, ys, yerr=sds, alpha=1.0, fmt=None, ecolor='red', zorder=9)	

	#plt.set_xlabel("Time")
	#plt.set_ylabel("Distance")
	
	
	#if re.search(r"x", args['--log']): plt.set_xscale('log')
	#if re.search(r"y", args['--log']): plt.set_yscale('log')

	#fig.savefig(PLOT_DIR+method+".pdf")

#print "# Done plotting."

#"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#Print the hypotheses
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#"""

#Hform2constants = defaultdict(list) # a list of lists of constants for each structural form
#Hform2best = dict() # The best hypothesis for each structural form
#o = open(DIRECTORY+"/hypotheses-postprocess.txt", 'w')

#print >>o, "i first.found sample.count lp lpZ prior ll h hcon hform f0 f1 nconstants "+" ".join(['C%i'%i for i in xrange(MAX_CONSTANTS)])
#for hi, h in enumerate( sorted(H2cnt.keys(), key=lambda x: H2post[x][0], reverse=True) ):
	## And convert H to a more reasonable form:
	#try:
		#sympystr = str(parse_expr(h))
	#except OverflowError: sympystr = "<OVERFLOW>"
	#except ValueError:    sympystr = "<VALUE_ERROR>"
	#except RuntimeError:  sympystr = "<RUNTIME_ERROR>"
	#except MemoryError:   sympystr = "<MEMORY_ERROR>"
	
	## And store the constants
	#constants = [ "NA" ] * MAX_CONSTANTS
	#nconstants = 0
	#for i,m in enumerate(re.findall(FPregex, sympystr)): 
		#nconstants += 1
		#if i < MAX_CONSTANTS: constants[i]=float(m) # only store the first this many
	
	## Now print out a modified format for easier processing
	#structural_form = re.sub(FPregex, 'C', sympystr) # The structural form, with no constants
	
	## Store these as constants for this form
	#Hform2constants[structural_form].append(constants)
	
	## Store the best value for this form
	#if structural_form not in Hform2best or H2post[h][0] > Hform2best[structural_form]:
		#Hform2best[structural_form] = H2post[h][0]
	
	#try: f0 = H2f[h](0.0)
	#except: f0 = "NA"
	#try: f1 = H2f[h](1.0)
	#except: f1 = "NA"
	
	## Print out this hypothesis
	#print >>o, hi, H2firstfound[h], H2cnt[h], H2post[h][0], H2post[h][0]-Z, H2post[h][1], H2post[h][2], q(h), q(sympystr), q(structural_form), f0,f1, nconstants, ' '.join(map(str,constants))	
#o.close()


#"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#Histograms of each constant
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#"""

#constantplotpath = DIRECTORY+"/constants/"
#if not os.path.exists(constantplotpath): os.mkdir(constantplotpath)

#for r, form in enumerate(sorted(Hform2best.keys(), key=lambda x: Hform2best[x], reverse=True)):
	
	## Make for the rth rank
	#if not os.path.exists(constantplotpath+"/%i/"%r): os.mkdir(constantplotpath+"/%i/"%r )
	
	## Now plot each constant
	#for ci in range(MAX_CONSTANTS):
		#if Hform2constants[form][0][ci] is "NA": break # don't do anything
		
		#fig = matplotlib.pyplot.figure(figsize=(4,4))
		#plt = fig.add_subplot(1,1,1)
		
		#hist, bins = numpy.histogram(map(lambda x: x[ci], Hform2constants[form]), bins=50)
		#width = 0.7 * (bins[1] - bins[0])
		#center = (bins[:-1] + bins[1:]) / 2
		#plt.bar(center, hist, align='center', width=width)
		
		#fig.savefig(constantplotpath+"/%i/%i.pdf"%(r,ci))
			
			
	## Only analyze this far
	#if r > 10: break 

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Plot the stats for comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





TODO: MAKE THIS PLOT, EVAL AVG FOR GP, SELECT POLYNOMIAL VIA AIC, and PRINT OUT EVERYTHING!


"""

o = open(DIRECTORY+"/gpuropolis-eval.txt", 'w')

for testtype in ['train', 'heldout']: # test on these
	
	
	# What data set do we train on?
	# NOTE: For our hypotheses, this is fixed by the GPU mcmc, NOT here
	if traintype=='all':          train_indicator = numpy.array( [True]*LEN )
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
	
	if not any(test_indicator): continue

	# ####################################
	## Compute the polynomial fits:
	best_poly, best_poly_AIC = None, float("inf")
	for degree in xrange(0,6):	
		coefs = numpy.polyfit(trainx, trainy, degree, w=1.0/(trainsds*trainsds)) ## inverse variance weighting polynomial fit
		
		p = numpy.poly1d(coefs)
		myll =  data_ll( p(testx), testy, testsds)
		aic = 2*(degree+1) - 2*myll
		
		if aic < best_poly_AIC:
			best_poly = degree, p, myll
			best_poly_AIC = aic
		
		
		# And evaluate on the test
		print traintype, testtype, "P.%i.LL"%degree, myll
	print traintype, testtype, "P.AIC", best_poly[2]
	
	
	# ####################################
	# Compute for the model
	# The expected ll averaging via counts
	EcntLL = 0.0
	for h in H2cnt:	
		EcntLL += float(H2cnt[h]) * data_ll( H2f[h](testx), testy, testsds)
	EcntLL /= float(Zcnt)
	print traintype, testtype, "E.cnt.LL", EcntLL


	## The ll
	print traintype, testtype, "MAP.LL", data_ll(H2f[map_h](testx), testy, testsds)
	
	# ####################################
	# Compute Gaussian process
	# From http://scikit-learn.org/0.11/auto_examples/gaussian_process/plot_gp_regression.html
	# The nugget must have (sd/y)**2 for the squared exponential, and we recover the data correctly!
	gp = GaussianProcess(corr='squared_exponential', theta0=1e-1,thetaL=1e-3, thetaU=1, nugget=(trainsds / trainy)**2.0, random_start=100)
	gp.fit(  numpy.atleast_2d(trainx).T,  numpy.atleast_2d(trainy).T )
	
	# Make the prediction on the meshed x-axis (ask for MSE as well)
	y_pred, MSE = gp.predict(  numpy.atleast_2d(testx).T , eval_MSE=True)
	sigmas = numpy.sqrt(MSE)
	print traintype, testtype, "GP.ML", data_ll( y_pred, testy, testsds)
	
	## And do the GP integrating over it's (Gaussian) average predictions
	# TODO: DO THIS!!
	#print traintype, testtype, "GP.ML", data_ll( y_pred, testy, testsds)
	
	
	## TODO: THE GAUSSIAN PROCESS SHOULD RESPECT THE NOISE IN THE DATA MESUREMENT WHEN COMPUTING LIKELIHOOD
	## SO LL SHOULD TAKE A estimate_error term!!
	
	
	# ####################################
	# Plot the data
	fig = matplotlib.pyplot.figure(figsize=(5,4))
	plt = fig.add_subplot(1,1,1)
	plt.scatter(xs, ys, label=u'Data', s=15, marker=".", color='red', zorder=10)
	plt.errorbar(xs, ys, sds, fmt='r.', markersize=0, linewidth=0.2, label=u'Observations', zorder=10)
	
	ERROR_ALPHA = 0.4
	LINEWIDTH   = 1.2
	
	# Do GP plot
	y_pred, MSE = y_pred, MSE = gp.predict(  numpy.atleast_2d(newx).T , eval_MSE=True)
	sigmas = numpy.sqrt(MSE)
	plt.plot(  newx, y_pred, 'b-', label=u'Prediction', lw=LINEWIDTH, zorder=1)
	plt.fill_between(newx, y_pred - 1.9600 * sigmas,(y_pred + 1.9600 *  sigmas), alpha=ERROR_ALPHA, color='blue', edgecolor='None', zorder=1)

	# Do polynomial plot (best AIC)
	plt.plot( newx, best_poly[1](newx), color='orange', lw=LINEWIDTH, zorder=2)
	
	# Do avg hypothesis plot -- using quantiles
	curves = []
	for nx in newx:
		newy = numpy.array([ H2f[h](nx) for h in H2f.keys()] )
		curves.append(scipy.stats.mstats.mquantiles(newy, prob=[0.025, 0.5, 0.975]))
	curves = numpy.array(curves)
	plt.plot(newx, curves[:,1], color='green', linewidth=LINEWIDTH, zorder=3)
	plt.fill_between(newx, curves[:,0], curves[:,2], color='green', alpha=ERROR_ALPHA, edgecolor='None', zorder=3)
	
	# And plot the MAP
	plt.plot(newx, numpy.array([H2f[map_h](nx) for x in newx]), color='green', linewidth=LINEWIDTH, linestyle=':')
	
	
	# Finish up plotting
	plt.set_xlim( *smartrange(xs) )
	plt.set_ylim( *smartrange(ys, sds=sds)  )
	fig.savefig("o%s.pdf"%testtype)        
	 
	
	
o.close()
