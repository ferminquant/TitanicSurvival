library(h2o)
library(parallel)
localH2O = h2o.init(nthreads=detectCores()-1)

data = read.csv("train.csv", header = TRUE)
data$Age = ifelse(is.na(data$Age), -1, data$Age)
hdata = as.h2o(data)
hdata$Survived = as.factor(hdata$Survived)
hdata = hdata[-c(1,4,9,11)]

hdata_split = h2o.splitFrame(hdata, ratios = c(0.1))
hdata_test = hdata_split[[1]]
hdata_train = hdata_split[[2]]

RF = h2o.randomForest(x=2:8,y=1,training_frame=hdata_train,
                      validation_frame=hdata_test,
                      nfolds=10)

h2o.shutdown(FALSE)