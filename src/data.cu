/*
 * GPUropolis - 2013 Aug 30 - Steve Piantadosi 
 * 
 * Functions and structs for manipulating data
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <vector>

using namespace std;

// what type is the program?
typedef float data_t;

// A struct for storing data
typedef struct datum {
	data_t input;
	data_t output;
	data_t sd; // stdev of the output|input. 
} datum;


// Load data froma  file, putting it into our structs.
// This allows us to trim our data if we want
vector<datum>* load_data_file(const char* datapath, int FIRST_HALF_DATA, int EVEN_HALF_DATA) {

	FILE* fp = fopen(datapath, "r");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << datapath <<"\n"; exit(1);}
	
	vector<datum>* d = new vector<datum>();
	char* line = NULL; size_t len=0; float x,y,sd; 
	while( getline(&line, &len, fp) != -1) {
		if( line[0] == '#' ) continue;  // skip comments
		else if (sscanf(line, "%f\t%f\t%f\n", &x, &y, &sd) == 3) { 
			d->push_back( (datum){.input=(data_t)x, .output=(data_t)y, .sd=(data_t)sd} );
		}
		else {
			cerr << "*** ERROR IN PARSING INPUT\t" << line << endl;
			exit(1);
		}
	}
	fclose(fp);
	
	// Trim the data based on first/second half or even odd
	if(FIRST_HALF_DATA){
		int mid = d->size()/2;
		for(int i=d->size()-1;i>=mid;i--) {
			d->erase(d->begin()+i);
		}
	}
	if(EVEN_HALF_DATA) {
		for(int i=d->size()-1;i>=0;i-=2) {
			d->erase(d->begin()+i);
		}
	}
		
	return d;
}