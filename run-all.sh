 
ITERATIONS=100000
OUTER_BLOCKS=100
N=8096
OUTROOT=run/
BURN=0

# The executable
EXEC=./gpuropolis

# Make globs work how we want. Linux is magic.
shopt -s globstar

OUTROOT=./out

# WHICHHALF=all
# for DATA in $(ls -d data-sources/Regression/1); do

# DATA=data-sources/Regression/-10_20
# DATA=data-sources/Science/BalmerSeries
# for WHICHHALF in 'all' 'first-half' 'even-half' ; do
# for DATA in $(ls -d data-sources/Regression/*); do
for DATA in $(ls -d data-sources/NIST/*) $(ls -d data-sources/Regression/*) ; do
# for DATA in $(ls -d data-sources/Regression/60_100); do
# # for DATA in $(ls -d data-sources/Science/COBE); do
# for DATA in $(ls -d data-sources/Science/*) ; do
# for DATA in $(ls -d data-sources/Science/*)  $(ls -d data-sources/Stats/*) $(ls -d data-sources/Polynomial/*)  $(ls -d data-sources/NIST/*)  $(ls -d data-sources/Regression/*) ; do
# for DATA in $(ls -d data-sources/Science/*)  $(ls -d data-sources/Stats/*) $(ls -d data-sources/NIST/*) ; do

	echo Running $DATA 
	
	OUT=$OUTROOT/$DATA/
	
	rm -rf $OUT
	
	mkdir -p $OUT
	mkdir -p $OUT/used-data/
	cp -r $DATA/* $OUT/
	
	# run the CUDA MCMC; must use gnu time in order to output
	# But here we also run time so it prints on command line too
	time /usr/bin/time --output=$OUT/time.txt $EXEC --steps=$ITERATIONS --in=$DATA/data.txt --outer=$OUTER_BLOCKS --N=$N --out=$OUT
	
	sort -g -k3 --parallel=8 $OUT/samples.txt | tail -n 10000 > $OUT/tops.txt &
	
	# And a python post-processing script to do plotting.
	# Run in the background so we can move to the next plot
# 	nice -n 19 python postprocess.py --directory=$OUT &

done
# done
