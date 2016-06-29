 
ITERATIONS=100 #200 #50000 # K20: 10000 #200000   ## About 1min for each 1k, for Zipf
OUTER_BLOCKS=100 #100 #100 outer / 100 burn works really well!  
N=50000
OUTROOT=run/
BURN=100 #100
ENUMERATION=9

# The executable
EXEC=./gpumcmc

# Make globs work how we want. Linux is magic.
shopt -s globstar

OUTROOT=./run

WHICHHALF=all

# for DATA in $(ls -d data-sources/Regression/1); do
# DATA=data-sources/Regression/-10_20
# for DATA in $(ls -d data-sources/Regression/*); do
# DATA=data-sources/Science/COBE
# DATA=data-sources/Stats/Logistic
# DATA=data-sources/NIST/MGH09
# for WHICHHALF in 'first-half' 'even-half' 'all' ; do
for DATA in data-sources/Science/COBE data-sources/Science/Hubble data-sources/Science/Galileo ; do
# for DATA in $(ls -d data-sources/NIST/*) ; do
# for DATA in $(ls -d data-sources/Science/*)  $(ls -d data-sources/Stats/*) $(ls -d data-sources/NIST/*) ; do

	echo Running $DATA 
	
	OUT=$OUTROOT/$DATA-$WHICHHALF/
	
	rm -rf $OUT
	
	mkdir -p $OUT
	mkdir -p $OUT/used-data/
	cp -r $DATA/* $OUT/used-data/
	
	# run the CUDA MCMC; must use gnu time in order to output
	# But here we also run time so it prints on command line too
	time /usr/bin/time --output=$OUT/time.txt $EXEC --burn=$BURN --enumeration=$ENUMERATION --iterations=$ITERATIONS --in=$DATA/data.txt --outer=$OUTER_BLOCKS --burn=$BURN_BLOCKS --N=$N --out=$OUT --$WHICHHALF
	
	sort -g -k 3 --parallel=4 $OUT/samples.txt | tail -n 10000 > $OUT/tops.txt &
	
done
# done
