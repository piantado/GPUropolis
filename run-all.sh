 

ITERATIONS=50000 # K20: 10000 #200000   ## About 1min for each 1k, for Zipf
BURN_BLOCKS=0 ## Don't need to burn if taking maps and sampling. 
OUTER_BLOCKS=100 #20  
N=100000 #50000 # 131072 #131072 # 131072 # 128*1024
OUTROOT=run/

EXEC=./gpumcmc

# Make globs work how we want
shopt -s globstar

OUTROOT=./run

WHICHHALF=all
DATA=data-sources/Science/BalmerSeries
# for WHICHHALF in 'all' 'first-half' 'even-half' ; do
# for DATA in $(ls -d data-sources/Science/*)  $(ls -d data-sources/Stats/*) $(ls -d data-sources/NIST/*) ; do
	
	echo Running $DATA !/
	
	OUT=$OUTROOT/$DATA-$WHICHHALF/
	
	rm -rf $OUT
	
	mkdir -p $OUT
	mkdir -p $OUT/used-data/
	cp -r $DATA/* $OUT/used-data/
	
	# run the CUDA MCMC; must use gnu time in order to output
	# But here we also run time so it prints on command line too
	time /usr/bin/time --output=$OUT/time.txt $EXEC --iterations=$ITERATIONS --in=$DATA/data.txt --outer=$OUTER_BLOCKS --burn=$BURN_BLOCKS --N=$N --out=$OUT --$WHICHHALF
	
	# And a python post-processing script to do plotting.
	# Run in the background so we can move to the next plot
	nice -n 19 python postprocess.py --directory=$OUT &

# done
# done