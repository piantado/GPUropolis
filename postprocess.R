
library(Hmisc)
library(scales) # for adding alpha 


# DIRECTORY="out/data-sources/NIST//"
# DIRECTORY="out/data-sources/Science/BalmerSeries/"
# DIRECTORY="run/data-sources/Science/Galileo-/"
# DIRECTORY="run/data-sources/Science/Fibonacci-/"
DIRECTORY="out/data-sources/Stats/Constant-10/"
# DIRECTORY="out/"
# DIRECTORY="run/data-sources/Regression/60_100-/"

d <- read.table(paste(DIRECTORY, "tops.txt", sep="/"), sep="\t")
# d <- read.table(paste(DIRECTORY, "samples.txt", sep="/"), sep="\t")
names(d) <- c("thread", "outer", "posterior", "prior", "likelihood", "h.struct", "h")


CUTOFF <- 5

data <- read.table(paste(DIRECTORY, "data.txt", sep="/"))


x <- seq(min(data$V1), max(data$V1), length.out=1000)

## Plot the data
# bitmap("output.png", height=6, width=6, res=400)
# postscript("o.pdf", height=6, width=6)
plot(data$V1, data$V2, col=4)
errbar(data$V1, data$V2, data$V2-data$V3, data$V2+data$V3, col=4)

dplot <- subset(d, posterior >  max(d$posterior) - CUTOFF)
for(r in 1:nrow(dplot)) {
#     print(dplot[r,])
    
    print(as.character(dplot[r,"h"]))
    print(dplot[r,])
    y <- eval(parse(text=as.character(dplot[r,"h"])))
#     f <- eval(parse(text=paste("function(x) {",as.character(dplot[r,"h"]), "}")))
    #ll <- sum(dnorm(f(data$V1)-data$V2, log=T)) # if we want this guy's likelihood
    
    # in case its constant
    if(length(y) == 1){
        y <- rep(y, length(x))
    }
    
    lines(x,y)
#     lines(x,y, color=rgb(0,0,0,alpha=0.5))
    
#     readline()
                      
}


# dev.off()





