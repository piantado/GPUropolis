
	
// 	for(int i=find_close_backwards(to->program, dMAX_PROGRAM_LENGTH-1);i<dMAX_PROGRAM_LENGTH;i++){
// 		cuPrintf("\tto %d %d \n", i, to->program[i]);
// 	}
// 	cuPrintf("------------\n");
// 	
	
// 	TODO: THIS OLD IMPLEMENTATION BELOW WAS FASTER AND PROBABLY INCORRECT
	
	
// 	      ABOVE IS SLOW BUT CLEAR AND SHOULD BE UPDATED
// 	int pos = endsplice;
// 	int start_ar = dMAX_PROGRAM_LENGTH - from->program_length;
// 	int xstart   = dMAX_PROGRAM_LENGTH - x->program_length;
// 	
// // 	set ar[start_ar...(start_ar+len_ar)] = x[MAX_PROGRAM_LENGTH-len_x:MAX_PROGRAM_LENGTH-1];
// // // 	nonincludsive of end_ar, but inclusive of start_ar. inclusive of start_x
// // 	NOTE: WE can get garbage on the left of our string if we insert soemthing in 0th (rightmost) position
// 
// // 	correct for the mis-alignment of x and the gap of what its replacing
// 	int shift = x->program_length - (pos+1-start_ar); 
// 	
// 	int xi = xstart;
// 	for(int ari=0;ari<dMAX_PROGRAM_LENGTH;ari++) {
// 		int in_splice_region = (ari>=start_ar-shift) && (ari<=pos);
// 		int in_final_region = (ari > pos);
// 		
// // 		wrap this for in case it goes off the end
// 		int ar_ari_shift = ifthen( (ari+shift < 0) || (ari+shift >= dMAX_PROGRAM_LENGTH), 0, from->program[ari+shift]);
// 		
// 		to->program[ari] = ifthen(in_splice_region, x->program[xi], ifthen(in_final_region, from->program[ari], ar_ari_shift) );
// 		
// 		xi += in_splice_region; // when in the splice region, increment by 1
// 	}
// 	
// // 	and update the length! This is necessary!
// 	to->program_length = from->program_length + shift;
