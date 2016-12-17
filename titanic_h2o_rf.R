library(h2o)
library(parallel)
localH2O = h2o.init(nthreads=detectCores()-1)

data = read.csv("train.csv", header = TRUE)
data$Age = ifelse(is.na(data$Age), -1, data$Age)
hdata = as.h2o(data)
hdata$Survived = as.factor(hdata$Survived)

hdata_split = h2o.splitFrame(hdata, ratios = c(0.1))
hdata_test = hdata_split[[1]]
hdata_train = hdata_split[[2]]



#RF1 = h2o.randomForest(x = 1:30, y = 34, d1.train, validation = d1.cv)

h2o.shutdown(FALSE)