
import re
import numpy

import sympy
from sympy import init_printing, Symbol, expand
from sympy.parsing.sympy_parser import parse_expr
from collections import defaultdict
 
MAX_CONSTANTS = 10 # Max number of constants we'll display

# How to match floating points. Shoot me now.
FPregex = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?' 

"""
---------------------------------------------------------------------------------------
	For loading our data files
---------------------------------------------------------------------------------------
"""

def load_data(f):
	"""
		Loads a data file, returning an x,y,sd triple. 
		Takes a "type" specifying:
		
	"""
	xs, ys, sds = [], [], []
	for l in open(f, 'r'):
		l = l.strip()
		if re.match("\s*#",l): continue
		
		x,y,sd = map(float, re.split("\s+", l))
		
		xs.append(x)
		ys.append(y)
		sds.append(sd)
	xs = numpy.array(xs)
	ys = numpy.array(ys)
	sds = numpy.array(sds)

	return xs,ys,sds

"""
---------------------------------------------------------------------------------------
	For loading hypotheses
---------------------------------------------------------------------------------------
"""	

def load_hypotheses(f, trim=10.0):
	trim = float(trim) # Just in case!
	
	fullMaxLP = float("-inf")
	for l in open(f, 'r'):
		if re.match("\s*#",l): continue
		parts = re.split("\t", l)
		lp = float(parts[2])
		if lp>fullMaxLP and not isnan(lp): fullMaxLP = lp
		
	print "# Max posterior found:", fullMaxLP

	H2cnt = defaultdict(int)
	H2post = dict()
	H2f = dict()
	H2firstfound = dict() # When did we first find this hypotheiss?
	
	sn = -1 # what sample number are we on?
	for l in open(f, 'r'):
		if re.match("\s*#",l): continue
		l = l.strip()
		parts = re.split("\t", l)
		
		rank, n, lp, prior, ll, h = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), parts[-1]
		lp = float(lp)
		
		# increment the sample number -- no longer is this reported explicitly
		if rank==0: sn += 1
		
		# Fully skip these -- not just in plotting. If they are too terrible
		if isnan(lp) or fullMaxLP-lp>trim: continue
		
		h = re.sub("\"", "", h)
		
		if h not in H2cnt and not isnan(lp):
			
			## Store some things:
			H2cnt[h] += 1
			H2post[h] = map(float, [lp, prior, ll])
			H2firstfound[h] = sn
			
			# Keep track of the MAP hypothesis
			
			try:
				H2f[h] = f = eval('lambda x: '+h)
			except MemoryError:
				H2f[h] = lambda x: float("nan")

	return H2cnt, H2post, H2f, H2firstfound

"""
---------------------------------------------------------------------------------------
	For defining a bunch of math functions:
---------------------------------------------------------------------------------------
"""

from math import isinf
from numpy import exp, sin as sin_, arcsin as asin_, isnan, log as log_, power as power_, sqrt as sqrt_
from scipy.special import gamma as gamma_

try:			from scipy.misc import logsumexp 
except ImportError:	from scipy.maxentropy import logsumexp 

def nanwrap(fn):
	def wrapped(*args):
		if numpy.isnan(args).any(): return float("nan")
		return fn(*args)
	return wrapped

def q(x):
	return "\""+str(x)+"\""

@nanwrap
def sqrt(x):
	return sqrt_(x)

@nanwrap
def sin(x): 
	if abs(x)==float("inf"): return float("nan")
	return sin_(x)

@nanwrap
def asin(x): return asin_(x)

@nanwrap
def add(x,y): return x+y

@nanwrap
def sub(x,y): return x-y

@nanwrap
def mul(x,y): return x*y


@nanwrap
def sgn(x): 
	if x<0: return -1
	if x>0: return 1
	return 0

@nanwrap
def div(x,y): 
	if y == 0: return x*float("inf")
	return x/y

@nanwrap
def gamma(x):
	if(x <= 0): return float("nan")
	
	return gamma_(x)

@nanwrap
def log(x): 
	return numpy.log(x)
	#if(x == 0): return float("-inf")
	#if(x < 0): return float("nan")
	#return log_(x)

# Define a version that matches CUDA weirdness
def power(x,y):
	if isnan(y) and x == 1.0: return 1.0
	if isnan(x) and y == 0.0: return 1.0
	if (x<0) and (x != float("-inf")) and round(y) != y: return float("nan")
	try:
		ret = power_(x,y)
	except OverflowError: return float("inf")*x
	return ret
	
	
# Maps that can handle failure
# and that does not return inf/nan
def failmap(f, xs):
	# A better mapping that deals with some failures
	ys = []
	newx = []
	for x in xs:
		y = None
		try: 
			y = f(x)
		except: pass
		
		if y is not None and not isinf(y) and not isnan(y): 
			ys.append(y)
			newx.append(x)
		
		
	return newx, ys	

def smartrange(v, sds=0.0, pad=1.1):
	
	if(len(v)==0): return [-1,1]
		
	r = max(v+sds)-min(v-sds)
	padr = pad * r
	return min(v-sds)-(padr-r)/2., max(v+sds)+(padr-r)/2.

def ll(x,y,sd):
	return numpy.sum( scipy.stats.norm.logpdf( numpy.abs(x - y)/sd))
