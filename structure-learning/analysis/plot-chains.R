

library(ggplot2)

d <- read.table("run/data-sources/Science/BalmerSeries-all/samples.txt")
d$nlposterior <- -d$V5

p <- ggplot( data=subset(d,V4<100), aes(x=V2, y=nlposterior, color=V4, group=V4)) + geom_line() + scale_y_log10()
# p <- ggplot( data=d, aes(x=V2, y=nlposterior, color=V4, group=V4)) + geom_line() + scale_y_log10()
# p <- ggplot( data=d, aes(x=V2, y=nlposterior, color=V4, group=V4)) + geom_line() + scale_y_log10()
# p

ggsave(filename="plot.pdf", plot=p)

# print(min(d[d$V2==max(d$V2),"nlposterior"]))
print(median(d[d$V2==max(d$V2),"nlposterior"]))

# d[d$V4==1822,]

