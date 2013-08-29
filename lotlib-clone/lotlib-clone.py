"""
	A clone of the program used to do symbolic regression on the GPU
"""
import LOTlib.MetropolisHastings
from LOTlib.PCFG import *
from LOTlib.Hypothesis import GaussianLOTHypothesis

PRIOR_TEMPERATURE = 1.0
LL_SD = 1.0

STEPS=100
SKIP=1000

#data = "data-sources/COBE/data.txt"
data = []

G = PCFG()
G.add_rule('START', '', ['EXPR'], 1.0)

G.add_rule('EXPR', '1', [], 0.35)
G.add_rule('EXPR', 'x', [], 0.05) # these terminals should have None for their function type; the literals

p2 = 0.3 # How often do we propose 2-args?
G.add_rule('EXPR', 'ADD', ['EXPR', 'EXPR'], p2/5.)
G.add_rule('EXPR', 'SUB', ['EXPR', 'EXPR'], p2/5.)
G.add_rule('EXPR', 'MUL', ['EXPR', 'EXPR'], p2/5.)
G.add_rule('EXPR', 'DIV', ['EXPR', 'EXPR'], p2/5.)
G.add_rule('EXPR', 'POW', ['EXPR', 'EXPR'], p2/5.) # including this gives lots of overflow

p1 = 0.3 # How often do we propose 1-args?
G.add_rule('EXPR', 'NEG', ['EXPR'], p1/5.) # including this gives lots of overflow
G.add_rule('EXPR', 'LOG', ['EXPR'], p1/5.)
G.add_rule('EXPR', 'EXP', ['EXPR'], p1/5.)
G.add_rule('EXPR', 'SIN', ['EXPR'], p1/5.)
G.add_rule('EXPR', 'ASIN', ['EXPR'], p1/5.)

initial_hyp = GaussianLOTHypothesis(G, prior_temperature=PRIOR_TEMPERATURE, ll_sd=LL_SD)
for x in LOTlib.MetropolisHastings.mh_sample(initial_hyp, data, STEPS, skip=SKIP, trace=F):
	print x.lp, x