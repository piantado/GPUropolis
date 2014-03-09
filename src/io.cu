/*
	Input/output routines on hypotheses
*/

const int MAX_OP_LENGTH = 256; // how much does each string add at most?
char SS[MAX_PROGRAM_LENGTH*2][MAX_OP_LENGTH*MAX_PROGRAM_LENGTH]; 

void print_program_as_expression(FILE* fp, hypothesis* h) {
	/*
	 * Print things nicely on displays.
	 * Currently these are set to play nice with sympy for processing output
	 */
	
	char buf[MAX_PROGRAM_LENGTH*MAX_PROGRAM_LENGTH];
	
	int top = MAX_PROGRAM_LENGTH; // top of the stack
	int constant_i = 0; // what constant are we pointing at?
	int program_start = 0;
	
	// re-initialize our buffer
	for(int r=0;r<MAX_PROGRAM_LENGTH*2;r++) strcpy(SS[r], "0"); // since everything initializes to 0
	
	for(int p=program_start;p<MAX_PROGRAM_LENGTH;p++) { // NEED to start at the same place as virtual-machine (either 0 or program_start), or else consts can get out of sync!
		op_t op = h->program[p];
		
		switch(op) {
			case NOOP_:
				break;
			case ZERO_: 
				top += 1;
				strcpy(SS[top], "0"); // TODO: Shoudl ZERO_ And ONE_ be floats? Sympy doesn't fully simplify if floats
				break;
			case ONE_: 
				top += 1;
				strcpy(SS[top], "1");
				break;
			case X_:
				top += 1;
				strcpy(SS[top], "x");
				break;
			case ADD_:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, "+");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case SUB_:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, "-");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case MUL_:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, "*");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
				
			case DIV_:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, "/");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;
			case POW_:
				strcpy(buf, "(");
				strcat(buf, SS[top]);
				strcat(buf, "**");
				strcat(buf, SS[top-1]);
				strcat(buf, ")");
				top -= 1;
				strcpy(SS[top], buf);
				break;	
			case NEG_:
				strcpy(buf, "(-");
				strcat(buf, SS[top]);
				strcat(buf, ")");
				strcpy(SS[top], buf);
				break;
			case CONSTANT_:
				top += 1;			
				sprintf(SS[top],"%f",  (constant_i < MAX_CONSTANTS) * h->constants[constant_i]);
				constant_i += (constant_i < MAX_CONSTANTS);
				break;
			default: // Defaultly just use the name
				strcpy(buf, NAMES[op]);
				if(hNARGS[op]>=1){
					strcat(buf, "(");
					strcat(buf, SS[top]);
					for(int k=1;k<hNARGS[op];k++) { // append on all the arguments
						strcat(buf, ",");
						strcat(buf, SS[top-k]);
					}
					strcat(buf, ")");
				}
				else { // if we are a constant, stack increases
					top++;
				}
				strcpy(SS[top], buf);
				break;
				
		}
	}
	
	fprintf(fp, "%s", SS[top]);
}



void dump_to_file(const char* path, hypothesis* ar, int repn, int outern, int N, int append) {
	
	
	FILE* fp;
	if(append) fp = fopen(path, "a");
	else       fp = fopen(path, "w");
	if(fp==NULL) { cerr << "*** ERROR: Cannot open file:\t" << path <<"\n"; exit(1);}
	
	for(int n=0;n<N;n++) {
		hypothesis* h = &ar[n];
		fprintf(fp, "%i\t%i\t%d\t%.3f\t%.3f\t%.3f\t%.3f\t", 
			repn, 
			outern, 
			n, 
			h->posterior,  
			h->constant_prior,
			h->structure_prior,
			h->likelihood
			);
		
		//print out the program
		fprintf(fp,"\"");
		for(int i=0;i<MAX_PROGRAM_LENGTH;i++) fprintf(fp, "%d ", h->program[i]);
		fprintf(fp,"\"\t");

		// print out constants
		fprintf(fp,"");
		for(int i=0;i<MAX_CONSTANTS;i++) fprintf(fp, "%.3f\t", h->constants[i]);
		fprintf(fp,"\t");			

		// and the program itself
		fprintf(fp, "\"");
		print_program_as_expression(fp, h );
		fprintf(fp, "\"\n");
	}
	
	fclose(fp);	
}
