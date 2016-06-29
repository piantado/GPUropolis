

d <- read.table("samples.txt", header=F)

q <- subset(d, V10=="0 0 1 2 1 10 9 6 5 5 ")

plot(q$V11, q$V12)


