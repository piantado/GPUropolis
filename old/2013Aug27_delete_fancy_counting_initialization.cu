// -----------------------------------------------------------------------
	// initialize the tail of host_hypotheses
	// we do this by counting tmp_ops in base NUM_OPS 
	// -----------------------------------------------------------------------
	
// 	op_t tmp_ops[MAX_PROGRAM_LENGTH];
// 	for(int i=0;i<MAX_PROGRAM_LENGTH;i++) { tmp_ops[i] = 0x0; }
// 	
// 	int n=0;
// 	
// 	while(n<N) {
// 	
// 		// increment the tmp_ops
// 		int toi = MAX_PROGRAM_LENGTH-1; // start on rhs
// 		tmp_ops[toi]++;
// 		while(tmp_ops[toi] >= NUM_OPS) { // a carry loop 
// 			tmp_ops[toi] = 0x0; // zero out
// 			toi--; // move to one prior index
// 			tmp_ops[toi]++; // increment
// 		}
// 		
// 		// now see how we are
// 		int i=MAX_PROGRAM_LENGTH-1;
// 		int nopen=0;
// 		while(tmp_ops[i] != NOOP) nopen += stack_change(tmp_ops[i--]);  // go back while there's no NOOP
// 		
// 		// if we aren't closed, we can be searched as a hypothesis
// 		// TODO: WHAT ABOUT THE CONSTANT HYPOTHESES, e.g. "1"?
// 		if(nopen > 0){
// 			memcpy( (void*)&host_hypotheses[n].program, tmp_ops, sizeof(tmp_ops));
// 			host_hypotheses[n].mutable_end = i; // the last allowable position we can change
// 			// to print out successful hypotheses
// 			for(int i=0;i<MAX_PROGRAM_LENGTH;i++) cerr << int(tmp_ops[i]) << ",";
// 			cerr << endl;
// 			
// 			n++;
// 		}
// 	}
// 	
// 	
// 	
// 	return 1;