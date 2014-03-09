/*
	Functions for manipulating arrays of hypotheses (on the host)

*/

int map_index(hypothesis* ar, int len) {
	int idx;
	double bestval = 1.0/0.0;
	for(int i=0;i<len;i++){
		double post = ar[i].posterior;
		if( post > bestval && ! isnan(post)){
			bestval = post;
			idx = i;
		}
	}
	return idx;
}


double get_normalizer(hypothesis* ar, int len, double temperature){
	
	double best = ar[map_index(ar,len)].posterior / temperature;
	
	// Compute the normalizer:
	double Z = 0.0;
	for(int i=0;i<len-1;i++) {
		double post = ar[i].posterior / temperature;
		if(!isnan(post)) Z += exp(post - best);
	}
	Z += best;
	
	return Z;			
}

void multinomial_sample(int N, hypothesis* toar, hypothesis* fromar, int fromlen, double temperature) {
	// For speed, assumes sorted with best last
	
	double Z = get_normalizer(fromar, fromlen, temperature);
	
	for(int to=0;to<N;to++) {
		double p = (float)rand()/(float)RAND_MAX; // sample a random number
		for(int j=fromlen-1;j>=0;j--) { // see what bin it falls into
			double post = fromar[j].posterior / temperature; 
			if(isnan(post)){ continue; }
			
			p = p - exp( post - Z);
			
			if(p<=0.0) {
				memcpy( (void*)&(toar[to]), &(fromar[j]), sizeof(hypothesis));
				break;
			}
		}
	}
	
}