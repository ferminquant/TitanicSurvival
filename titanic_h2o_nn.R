library(h2o)
library(parallel)
#localH2O = h2o.init(nthreads=detectCores()-1)

set.seed(1320)
data = read.csv("train.csv", header = TRUE)
data$Age = ifelse(is.na(data$Age), -1, data$Age)
data = data[sample(NROW(data)),]

hdata = as.h2o(data)
hdata$Survived = as.factor(hdata$Survived)
hdata = hdata[-c(1,4,9,11)]

hdata_split = h2o.splitFrame(hdata, ratios = c(0.1))
hdata_test = hdata_split[[1]]
hdata_train = hdata_split[[2]]

nn = h2o.deeplearning(x = 2:8, y = 1, training_frame = hdata_train, 
                      validation_frame = hdata_test,
                      nfolds = 10,
                      hidden = c(6),
                      standardize = TRUE,
                      activation = 'Tanh',
                      epochs = 100,
                      seed = 1320,
                      shuffle_training_data = TRUE,
                      variable_importances = TRUE)

summary(nn)

#h2o.shutdown(FALSE)