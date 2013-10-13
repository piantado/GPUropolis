
import re
import numpy
 


"""
---------------------------------------------------------------------------------------
	For defining a bunch of math functions:
---------------------------------------------------------------------------------------
"""

from math import isinf
from math import exp, sin as sin_, asin as asin_, isnan, log as log_, pow as power_, gamma as gamma_, sqrt as sqrt_

try:			from scipy.misc import logsumexp 
except ImportError:	from scipy.maxentropy import logsumexp 

def nanwrap(fn):
	def wrapped(*args):
		for a in args:	
			if isnan(a): return float("nan")
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
	if(x == 0): return float("-inf")
	if(x < 0): return float("nan")
	return log_(x)

# Define a version that matches CUDA weirdness
def power(x,y):
	if isnan(y) and x == 1.0: return 1.0
	if isnan(x) and y == 0.0: return 1.0
	if (x<0) and (x != float("-inf")) and round(y) != y: return float("nan")
	try:
		ret = power_(x,y)
	except OverflowError: return float("inf")
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