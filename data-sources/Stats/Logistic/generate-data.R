N <- 25

x <- rnorm(N,mean=0,sd=10)

y <- 11.23 * x - 3.4 # some arbitrary logistic transformation

p <- 1./(1.+exp(-y))

b <- (runif(N) < p)*1 # binary coin flips

data <- data.frame(x=x,b=b,sd=0.1)
write.table(data, "data.txt", row.names=F, col.names=F)




