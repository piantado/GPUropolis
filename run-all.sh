 

ITERATIONS=100 # K20: 10000 #200000   ## About 1min for each 1k, for Zipf
BURN_BLOCKS=0 ## Don't need to burn if taking maps and sampling. 
OUTER_BLOCKS=100 #00
N=100000 #50000 # 131072 #131072 # 131072 # 128*1024
OUTROOT=run/
PROPOSAL=2

EXEC=./gpumcmc

# for WHICHHALF in '' --first-half; do
# for DATA in Boyle CLT Null Iris Iris-Function COBE PrimeCounting Zipf Constant Hubble Galileo BraheMars Stirling Shannon Sort; do
for DATA in COBE; do

	echo Running $DATA !
	
	OUT=$OUTROOT/$DATA/
	
	mkdir $OUT
	mkdir $OUT/used-data/
	cp -r data-sources/$DATA/* $OUT/used-data/
	
	# run the CUDA MCMC; must use gnu time in order to output
	# But here we also run time so it prints on command line too
	time /usr/bin/time --output=$OUT/time.txt $EXEC --proposal=$PROPOSAL --iterations=$ITERATIONS --in=data-sources/$DATA/data.txt --outer=$OUTER_BLOCKS --burn=$BURN_BLOCKS --N=$N --out=$OUT
	
	# And a python post-processing script to do plotting.
	# Run in the background so we can move to the next plot
# 	nice -n 19 python plot.py --directory=$OUT &

done
