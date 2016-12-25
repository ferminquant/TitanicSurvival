library(h2o)
library(parallel)
#localH2O = h2o.init(nthreads=detectCores()-1)

data = read.csv("train.csv", header = TRUE)
data$Age = ifelse(is.na(data$Age), -1, data$Age)
hdata = as.h2o(data)
hdata$Survived = as.factor(hdata$Survived)
hdata = hdata[-c(1,4,9,11)]

hdata_split = h2o.splitFrame(hdata, ratios = c(0.1))
hdata_test = hdata_split[[1]]
hdata_train = hdata_split[[2]]

acc_dist_test = NULL
acc_dist_cv = NULL
for (i in 1:100) {
 acc_dist_test[i] = 0
 acc_dist_cv[i] = 0
}

i = 0
min_err = 10000
min_cv_err = 10000

for (i in 1:1000) {
  rand_ntrees = sample(1:128,1)
  rand_max_depth = sample(1:128,1)
  rand_mtries = sample(1:7,1)
  RF = h2o.randomForest(x=2:8,y=1,training_frame=hdata_train,
                        validation_frame=hdata_test,
                        nfolds=10,
                        ntrees=rand_ntrees,
                        max_depth=rand_max_depth,
                        mtries=rand_mtries
                        )
  
  tmp = length(RF@model$scoring_history$validation_classification_error)
  err = RF@model$scoring_history$validation_classification_error[tmp]
  cv_err = as.numeric(RF@model$cross_validation_metrics_summary[3,1])
  
  j = round((1-err)*100,0)
  acc_dist_test[j] = acc_dist_test[j]+1
  
  j = round((1-cv_err)*100,0)
  acc_dist_cv[j] = acc_dist_cv[j]+1
  
  if (err < min_err) {
    min_err = err
    tmp_err_cv = cv_err
    bestRF = RF
    tmp_i = i
    tmp_ntrees = rand_ntrees
    tmp_max_depth = rand_max_depth
    tmp_mtries = rand_mtries
  }
  if (cv_err < min_cv_err) {
    min_cv_err = cv_err
    cv_err_test = err
    bestRFcv = RF
    cv_i = i
    cv_ntrees = rand_ntrees
    cv_max_depth = rand_max_depth
    cv_mtries = rand_mtries
  }
  print(sprintf("  Current params: %i | %f | %f | %i | %i | %i ", i, 1-err, 1-cv_err, rand_ntrees, rand_max_depth, rand_mtries))
  print(sprintf("Best test params: %i | %f | %f | %i | %i | %i ", tmp_i, 1-min_err, 1-tmp_err_cv, tmp_ntrees, tmp_max_depth, tmp_mtries))
  print(sprintf("  Best cv params: %i | %f | %f | %i | %i | %i ", cv_i, 1-cv_err_test, 1-min_cv_err, cv_ntrees, cv_max_depth, cv_mtries))
}
#h2o.shutdown(FALSE)