 

ITERATIONS=200000 ## About 1min for each 1k, for Zipf
BURN_BLOCKS=0 ## Don't need to burn if taking maps and sampling. 
OUTER_BLOCKS=100 #00
N=10240
OUTROOT=run/
PROPOSAL=2

EXEC=./gpumcmc

# for WHICHHALF in '' --first-half; do
# for DATA in Boyle CLT Null Iris Iris-Function COBE PrimeCounting Zipf Constant Hubble Galileo BraheMars; do
# for DATA in Hubble BraheMars ; do
# for DATA in Iris Iris-Function COBE ; do
for DATA in Hubble CLT PrimeCounting; do

	echo Running $DATA !
	
	OUT=$OUTROOT/$DATA/
	LOG=$OUT/log.txt
	
	mkdir $OUT
	mkdir $OUT/used-data/
	cp -r data-sources/$DATA/* $OUT/used-data/
	
	#echo to a log file
	echo Run started on $(date)           >> $LOG
	echo by $(whoami) on $(hostname)      >> $LOG
	echo Executable MD5: $(md5sum $EXEC)  >> $LOG
	echo                                  >> $LOG      
	echo ----------------------------     >> $LOG      
	echo -- PARAMETERS:                   >> $LOG
	echo ----------------------------     >> $LOG      
	echo ITERATIONS=$ITERATIONS           >> $LOG
	echo BURN=$BURN_BLOCKS                >> $LOG
	echo OUTER=$OUTER_BLOCKS              >> $LOG
	echo N=$N                             >> $LOG
	echo                                  >> $LOG      
	
	# run the CUDA MCMC; must use gnu time in order to output
	# But here we also run time so it prints on command line too
	time /usr/bin/time --output=$OUT/time.txt $EXEC --proposal=$PROPOSAL --iterations=$ITERATIONS --in=data-sources/$DATA/data.txt --outer=$OUTER_BLOCKS --burn=$BURN_BLOCKS --N=$N --out=$OUT
	echo CUDA mcmc completed on $(date) >> $LOG
	
	# And a python post-processing script to do plotting.
	# We'll do this on the tops
	nice -n 19 python plot.py --in=$OUT/tops.txt --data=data-sources/$DATA/data.txt --out=$OUT &
	echo Python completed on $(date) >> $LOG	

# done
done
