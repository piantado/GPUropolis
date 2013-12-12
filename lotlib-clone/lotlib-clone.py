"""
	A clone of the program used to do symbolic regression on the GPU.
	Some small differences, but basically the same. 
"""
import argparse
import LOTlib.MetropolisHastings
from LOTlib.PCFG import *
from LOTlib.DataAndObjects import FunctionData
from LOTlib.Hypothesis import GaussianLOTHypothesis
sys.path.append("..")
from Shared import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse our arguments
parser = argparse.ArgumentParser(description='Weber MCMC')
parser.add_argument('--in', dest='in', type=str, default='data.txt', nargs="?", help='The data file')
parser.add_argument('--out', dest='out', type=str, default='run', nargs="?", help='Output directory')

parser.add_argument('--steps', dest='steps', type=int, default=5000, nargs="?", help='Number of steps')
parser.add_argument('--skip', dest='skip', type=int, default=100, nargs="?", help='Skip in chains')
#parser.add_argument('--N', dest='N', type=int, default=1, nargs="?", help='Number of chains')

args = vars(parser.parse_args())

PRIOR_TEMPERATURE = 1.0
# LL_TEMPERATURE = 1.0
# ACCEPTANCE_TEMPERATURE

EXPECTED_LENGTH = 5
PRIOR_XtoCONSTANT = 0.1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Load and set up the data

data = [ FunctionData([x], y, ll_sd=sd) for x,y,sd in zip(*load_data(args['in'])) ]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define the grammar

P =  (1.0-1.0/EXPECTED_LENGTH)/3.0 # Helper variable for defining probs
P_0arg = 1.0-2.*P
P_X = P_0arg * PRIOR_XtoCONSTANT;
P_CONSTANT = P_0arg * (1.0-PRIOR_XtoCONSTANT) / 1. # Number of constants

G = PCFG()
G.add_rule('START', '', ['EXPR'], 1.0)

G.add_rule('EXPR', '1', [], P_CONSTANT)
G.add_rule('EXPR', 'x', [], P_X)

P2 = ((1.-1./EXPECTED_LENGTH)/3.) / 5. # put mass equally on all functions, regadless of arity
G.add_rule('EXPR', 'ADD', ['EXPR', 'EXPR'], P2)
G.add_rule('EXPR', 'SUB', ['EXPR', 'EXPR'], P2)
G.add_rule('EXPR', 'MUL', ['EXPR', 'EXPR'], P2)
G.add_rule('EXPR', 'DIV', ['EXPR', 'EXPR'], P2)
G.add_rule('EXPR', 'POW', ['EXPR', 'EXPR'], P2) # including this gives lots of overflow

P1 = ((1.-1./EXPECTED_LENGTH)/3.) / 3.
G.add_rule('EXPR', 'NEG', ['EXPR'], P1) # including this gives lots of overflow
G.add_rule('EXPR', 'LOG', ['EXPR'], P1)
G.add_rule('EXPR', 'EXP', ['EXPR'], P1)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Run

h0 = GaussianLOTHypothesis(G, prior_temperature=PRIOR_TEMPERATURE)
for x in LOTlib.MetropolisHastings.mh_sample(h0, data, args['steps'], skip=args['skip'], trace=F):
	print x.lp, x
	
	
	
	
	
