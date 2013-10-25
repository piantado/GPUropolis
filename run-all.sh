 

ITERATIONS=10000 # K20: 10000 #200000   ## About 1min for each 1k, for Zipf
BURN_BLOCKS=0 ## Don't need to burn if taking maps and sampling. 
OUTER_BLOCKS=20 #00
N=100000 #50000 # 131072 #131072 # 131072 # 128*1024
OUTROOT=run/
PROPOSAL=2

EXEC=./gpumcmc

# Make globs work how we want
shopt -s globstar

# for DATA in Boyle CLT Null Iris Iris-Function COBE PrimeCounting Zipf Constant Hubble Galileo BraheMars Stirling Shannon Sort; do
for WHICHHALF in 'all' 'first-half' 'even-half'; do
# for WHICHHALF in 'all'; do
for DATA in NIST/MGH09 NIST/BoxBOD NIST/Eckerle4 NIST/MGH10 NIST/Rat42 NIST/Rat43 NIST/Thurber NIST/Bennett5 ; do

	echo Running $DATA !
	
	OUT=$OUTROOT/$DATA-$WHICHHALF/
	
	mkdir -p $OUT
	mkdir -p $OUT/used-data/
	cp -r data-sources/$DATA/* $OUT/used-data/
	
	# run the CUDA MCMC; must use gnu time in order to output
	# But here we also run time so it prints on command line too
	time /usr/bin/time --output=$OUT/time.txt $EXEC --proposal=$PROPOSAL --iterations=$ITERATIONS --in=data-sources/$DATA/data.txt --outer=$OUTER_BLOCKS --burn=$BURN_BLOCKS --N=$N --out=$OUT --$WHICHHALF
	
	# And a python post-processing script to do plotting.
	# Run in the background so we can move to the next plot
# 	nice -n 19 python plot.py --directory=$OUT --traintype=$WHICHHALF &

done
done