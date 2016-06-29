/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Functions for manipulating hypotheses on the host
 */

// move all duplicates to the end, after sorting
// this always returns tofill number of hypotheses, filling with defaulthyp
// if we run out!
// Returns the number of unique items
int delete_duplicates(hypothesis* ar, int tofill, int maxlen, hypothesis* defaulthyp) {
	
	int unique_count = 0;
	for(int i=0,j=0;i<tofill;i++,j++) {
		if(j<maxlen) {
			// skip forward over everything identical
			// NOTE: BECAUSE we do hypothesis_structurally_identical, we only store the top of each structure
			// ignoring the differences in constants!
			while(j+1 < maxlen && hypothesis_structurally_identical(&ar[j], &ar[j+1])) {
				j++;
			}
			
			if(j!=i) {
				COPY_HYPOTHESIS( &ar[i], &ar[j] );
				unique_count++;
			}
		}
		else { // out of hyppotheses, so pad with blankhyp
			COPY_HYPOTHESIS(&ar[i], defaulthyp);
			
		}
	}
	
	return unique_count;
}


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

void dump_to_file(const char* path, hypothesis* ar, int repn, int outern, int N, int append) {
	
	
	FILE* fp;
	if(append) fp = fopen(path, "a");
	else       fp = fopen(path, "w");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << path <<"\n"; exit(1);}
	
	for(int n=0;n<N;n++) {
		hypothesis* h = &ar[n];
		fprintf(fp, "%i\t%i\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t", 
			repn, 
			outern, 
			n, 
			h->chain_index, 
			h->posterior,  
			h->prior,
			h->likelihood, 
			h->proposal_generation_lp,
			h->program_length);
		
		//print out the program
		fprintf(fp,"\"");
		for(int i=0;i<MAX_PROGRAM_LENGTH;i++) fprintf(fp, "%d ", h->program[i]);
		fprintf(fp,"\"\t");		
// 		
		// print out constant types
// 		fprintf(fp,"\"");
// // 		for(int i=0;i<MAX_CONSTANTS;i++) fprintf(fp, "%i ", h->constant_types[i]);
// 		fprintf(fp,"\"\t");

		// print out constants
		fprintf(fp,"");
		for(int i=0;i<MAX_CONSTANTS;i++) fprintf(fp, "%.3f\t", h->constants[i]);
		fprintf(fp,"\t");			
		
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