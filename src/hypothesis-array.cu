/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Functions for manipulating hypotheses on the host
 */

int map_index(hypothesis* ar, int len, double prior_temperature, double likelihood_temperature) {
	int idx;
	double bestval = -1.0/0.0;
	for(int i=0;i<len;i++){
		double post = ar[i].prior/prior_temperature + ar[i].likelihood / likelihood_temperature;
		if( post > bestval && ! isnan(post)){
			bestval = post;
			idx = i;
		}
	}
	return idx;
}


double get_normalizer(hypothesis* ar, int len, double prior_temperature, double likelihood_temperature){
	
	int k = map_index(ar,len, prior_temperature, likelihood_temperature);
	double best = ar[k].prior/prior_temperature + ar[k].likelihood / likelihood_temperature;
	
	// Compute the normalizer:
	double Z = 0.0;
	for(int i=0;i<len-1;i++) {
		double post = ar[i].prior/prior_temperature + ar[i].likelihood / likelihood_temperature;
		if(!isnan(post)) Z += exp(post - best);
	}
	Z += best;
	
	return Z;			
}

void multinomial_sample(int N, hypothesis* toar, hypothesis* fromar, int fromlen, double prior_temperature, double likelihood_temperature) {
	// For speed, assumes sorted with best last
	
	double Z = get_normalizer(fromar, fromlen, prior_temperature, likelihood_temperature);
	
	for(int to=0;to<N;to++) {
		double p = (float)rand()/(float)RAND_MAX; // sample a random number
		for(int j=fromlen-1;j>=0;j--) { // see what bin it falls into
			double post = fromar[j].prior/prior_temperature + fromar[j].likelihood / likelihood_temperature;
			if(isnan(post)){ continue; }
			
			p = p - exp( post - Z );
			
			if(p<=0.0 || j==0) {
				memcpy( (void*)&(toar[to]), &(fromar[j]), sizeof(hypothesis));
				break;
			}
		}
	}
	
}

void dump_to_file(const char* path, hypothesis* ar, int N, int append) {
	
	
	FILE* fp;
	if(append) fp = fopen(path, "a");
	else       fp = fopen(path, "w");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << path <<"\n"; exit(1);}
	
	for(int n=0;n<N;n++) {
		hypothesis* h = &ar[n];
		fprintf(fp, "%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t", n, h->chain_index, h->posterior,  h->prior, h->likelihood, h->acceptance_ratio, h->program_length);
		
		// print out the program
/*		fprintf(fp,"\"");
		for(int i=0;i<hMAX_PROGRAM_LENGTH;i++) fprintf(fp, "%d ", h->program[i]);
		fprintf(fp,"\"\t");	*/	
		
		// print out constant types
		for(int i=0;i<MAX_CONSTANTS;i++) fprintf(fp, "%i\t", h->constant_types[i]);
			
		fprintf(fp, "\"");
		print_program_as_expression(fp, h );
		fprintf(fp, "\"\n");
	}
	
	fclose(fp);	
}


// void save_state(const char* path, hypothesis* ar, int N) {
// 	
// 	FILE* fp = fopen
// 	if(append) fp = fopen(path, );
// 	else       fp = fopen(path, "w");
// 	
// 	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << path <<"\n"; exit(1);}
// }