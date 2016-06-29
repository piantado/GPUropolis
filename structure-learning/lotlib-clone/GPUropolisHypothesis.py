from LOTlib.Miscellaneous import *
from LOTlib.Hypotheses.GaussianLOTHypothesis import GaussianLOTHypothesis


X_DEPTH_PENALTY = 100.0 # extra penalty for X depth. 0 here gives PCFG generation probability prior
X_PENALTY = 10.0

class GPUropolisHypothesis(GaussianLOTHypothesis):
	"""
		Special penalization on "x" as a function of depth
	"""
	
	def compute_prior(self):
		
		if self.value.count_subnodes() > self.maxnodes: 
			self.prior = -Infinity
		else: 
			self.prior = self.value.log_probability() / self.prior_temperature
			
			# iterate through, getting depths too
			for n,d in self.value.iterdepth():
				if n.name == 'x': self.prior -= (X_PENALTY + d*X_DEPTH_PENALTY)
			
		
		self.posterior_score = self.prior + self.likelihood
			
		return self.prior