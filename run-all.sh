 
ITERATIONS=10000
OUTER_BLOCKS=100
N=1000
OUTROOT=run/
BURN=0

# The executable
EXEC=./gpumcmc

# Make globs work how we want. Linux is magic.
shopt -s globstar

OUTROOT=./run

# WHICHHALF=all
# for DATA in $(ls -d data-sources/Regression/1); do

# DATA=data-sources/Regression/-10_20
# DATA=data-sources/Science/BalmerSeries
# for WHICHHALF in 'all' 'first-half' 'even-half' ; do
# for DATA in $(ls -d data-sources/Regression/*); do
# for DATA in $(ls -d data-sources/Regression/60_100); do
for DATA in $(ls -d data-sources/Science/Galileo); do
# for DATA in $(ls -d data-sources/Science/*) ; do
# for DATA in $(ls -d data-sources/Science/*)  $(ls -d data-sources/Stats/*) $(ls -d data-sources/NIST/*) ; do
# for DATA in $(ls -d data-sources/Science/*)  $(ls -d data-sources/Stats/*) $(ls -d data-sources/NIST/*) ; do

	echo Running $DATA 
	
	OUT=$OUTROOT/$DATA-$WHICHHALF/
	
	rm -rf $OUT
	
	mkdir -p $OUT
	mkdir -p $OUT/used-data/
	cp -r $DATA/* $OUT/used-data/
	
	# run the CUDA MCMC; must use gnu time in order to output
	# But here we also run time so it prints on command line too
	time /usr/bin/time --output=$OUT/time.txt $EXEC --burn=$BURN --iterations=$ITERATIONS --in=$DATA/data.txt --outer=$OUTER_BLOCKS --burn=$BURN_BLOCKS --N=$N --out=$OUT --$WHICHHALF
	
	sort -g -k 4 --parallel=8 $OUT/samples.txt | tail -n 10000 > $OUT/tops.txt &
	
	# And a python post-processing script to do plotting.
	# Run in the background so we can move to the next plot
# 	nice -n 19 python postprocess.py --directory=$OUT &

done
# done
