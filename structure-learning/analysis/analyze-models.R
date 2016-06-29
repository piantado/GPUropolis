
# R script to compare heldout likelihood (and things) on a data file
# for alternative models that are more easily implemented in R!

library(mgcv) # For GAM
library(gptk) # for gaussian process

DIRECTORY = "/home/piantado/Desktop/mit/Projects/GPU-tMCMC/BoxBOD-all/"

d <- read.table(paste(DIRECTORY,"used-data", "data.txt", sep="/"))
names(d) <- c("x", "y", "sd")

for( traintype in c("all", "first-half", "even-half")) {
for( testtype  in c("all", "training", "held-out")) {

	train_idx <- NULL
	if( traintype == "all") { train_idx <- 1:nrow(d) }
	else if (traintype == "first-half") { train_idx <- 1:(nrow(d)/2) }
	else if (traintype == "even-half")  { train_idx <- (1:nrow(d)) %% 2==1 } # ugh deal with R's 1-indexing
	train <- d[train_idx,]
	
	test <- NULL
	if( testtype == "all")            { test <- d }
	else if (testtype == "training")  { test <- d[train_idx,] }
	else if (testtype == "held-out")  { test <- d[!train_idx,] } # ugh deal with R's 1-indexing

	###################################
	## Do GAM 
	## This uses default smoothing and the number of data points as maximum df
	###################################
	
	## TODO: MAKE THIS s(X)!!!!
# 	g <- gam( y ~ s(x), data=train, weight=1/(train$sd*train$sd), k=nrow(train))
# 	
# 	g.pred <- predict(g, newdata=test)
# 
# 	print(g.pred)

	###################################
	## Do gaussian process 
	## This uses default smoothing and the number of data points as maximum df
	###################################
	options = gpOptions()
	options$kern$comp = list("rbf", "white")
	g <- gpCreate( 1, 1, as.matrix(d$x), as.matrix(d$y), options )
	
	gpPlot(model, xTest, yPred, yVar, ylim = c(-3, 3), col = "black")
	
}
}