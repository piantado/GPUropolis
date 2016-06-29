library(stringr)
library(Hmisc) # for wtd.var

# Plot the results of running on the Regression data sets
# This gets the estimated slopes and intercepts of the linear posteriors, as well as
# the total probability mass of linear posteriors

D <- NULL
for(f in list.files("run/data-sources/Regression", no..=TRUE, full.names=T)) {
	
	m <- str_extract_all(f, "([0-9\\.]+)")
	angle <- as.numeric(m[[1]][[1]])
	N  <- as.numeric(m[[1]][[2]])
	
	filename <- paste(f, "hypotheses-postprocess.txt", sep="/")
	if(!file.exists(filename)) { next }
	
	d <- read.table(filename, header=T)
	
	# the linear subset
	dl <- subset(d, is.element(hform, c("C", "-x", "x", "C*x", "C*x + C", "C*x - C", "x + C", "x - C")))
	if(nrow(dl) == 0) { next } # TODO: THIS SHOULD BE BETTER HANLDED SO IT COUNTS AS ZEROS
	
	dl$slope <- NA
	dl[dl$hform=="C", "slope"]       <- 0.0
	dl[dl$hform=="-x", "slope"]      <- -1.0
	dl[dl$hform=="x", "slope"]       <- 1.0
	dl[dl$hform=="C*x", "slope"]     <- dl[dl$hform=="C*x", "C0"]
	dl[dl$hform=="x + C", "slope"]     <- 1.0
	dl[dl$hform=="x - C", "slope"]     <- 1.0
	dl[dl$hform=="C*x + C", "slope"] <- dl[dl$hform=="C*x + C", "C0"]
	dl[dl$hform=="C*x - C", "slope"] <- dl[dl$hform=="C*x - C", "C0"]
	
	dl$intercept <- NA
	dl[dl$hform=="C", "intercept"]       <- dl[dl$hform=="C", "C0"] 
	dl[dl$hform=="-x", "intercept"]      <- 0.0
	dl[dl$hform=="x", "intercept"]       <- 0.0
	dl[dl$hform=="C*x", "intercept"]     <- 0.0
	dl[dl$hform=="x + C", "intercept"] <- dl[dl$hform=="x + C", "C0"]
	dl[dl$hform=="x - C", "intercept"] <- dl[dl$hform=="x - C", "C0"]
	dl[dl$hform=="C*x + C", "intercept"] <- dl[dl$hform=="C*x + C", "C1"]
	dl[dl$hform=="C*x - C", "intercept"] <- -dl[dl$hform=="C*x - C", "C1"]
	
	# renormalizer here on these
	dlZ <- log(sum(exp(dl$lpZ)))
	dl$lpZ <- dl$lpZ - dlZ
	
	mslope     <- weighted.mean( dl$slope, exp(dl$lpZ) )
	mintercept <- weighted.mean( dl$intercept, exp(dl$lpZ) )
	
	if(exp(dlZ) < 0.75) {
		print(f)
		print(sort(table(d$hform)))
	}
# 	slope.sd <- sqrt(wtd.var( dl$slope, exp(dl$lpZ)))
	
	D <- rbind(D, data.frame( angle=angle,
				  N=N,
				  tslope=tan(angle * pi/180),
				  slope=mslope,
# 				  slope.sd=slope.sd,
				  intercept=mintercept,
				  lP=dlZ))
				  
}


par(mfrow=c(1,3))

# Plot the real vs estimated slopes for linear components
plot(D$tslope, D$slope, col=D$N, cex=2, pch=20)
# errbar(D$tslope, D$slope, D$slope-D$slope.sd, D$slope+D$slope.sd, add=T)
abline(0,1)

# Plot the real vs estimated slopes for linear components
plot(D$tslope, D$intercept, col=D$N, cex=2, pch=20, ylim=c(-1,1))
abline(0,0)


# how much probability mass is on linear?
plot(D$tslope, exp(D$lP), col=D$N, cex=2, pch=20, ylim=c(0,1))
abline(1,0)

