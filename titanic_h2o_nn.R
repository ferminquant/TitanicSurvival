library(h2o)
library(parallel)
#localH2O = h2o.init(nthreads=detectCores()-1)

#set.seed(1320)
data = read.csv("train.csv", header = TRUE)
data$Age = ifelse(is.na(data$Age), -1, data$Age)
data = data[sample(NROW(data)),]

hdata = as.h2o(data)
hdata$Survived = as.factor(hdata$Survived)
hdata = hdata[-c(1,4,9,11)]

hdata_split = h2o.splitFrame(hdata, ratios = c(0.1))
hdata_test = hdata_split[[1]]
hdata_train = hdata_split[[2]]

# nn = h2o.deeplearning(x = 2:8, y = 1, training_frame = hdata_train, 
#                       validation_frame = hdata_test,
#                       nfolds = 10,
#                       hidden = c(100,100,100,100,100),
#                       standardize = TRUE,
#                       activation = 'Tanh',
#                       epochs = 100,
#                       seed = 1320,
#                       shuffle_training_data = TRUE,
#                       variable_importances = TRUE)
#summary(nn)

#for hyperparameter optimization
models <- c()
tmp_min_err = 10000
#for (i in 1:100) {
while (TRUE) {
  rand_activation <- c("Tanh", "TanhWithDropout", "Rectifier","RectifierWithDropout", "Maxout", "MaxoutWithDropout")[sample(1:6,1)]
  rand_numlayers <- sample(2:5,1)
  rand_hidden <- c(sample(10:150,rand_numlayers,T))
  rand_l1 <- runif(1, 0, 1e-3)
  rand_l2 <- runif(1, 0, 1e-3)
  if (rand_activation == "TanhWithDropout" ||
      rand_activation == "RectifierWithDropout" ||
      rand_activation == "MaxoutWithDropout"){
    rand_dropout <- c(runif(rand_numlayers, 0, 0.6))
  }
  else {
    rand_dropout <- NULL
  }
  rand_input_dropout <- runif(1, 0, 0.5)
  print(i)
  print(rand_activation)
  print(rand_hidden)
  dlmodel <- h2o.deeplearning(x=2:8, y=1, training_frame = hdata_train, 
                              validation_frame = hdata_test, 
                              epochs=10,
                              activation=rand_activation, 
                              hidden=rand_hidden, 
                              l1=rand_l1, 
                              l2=rand_l2,
                              input_dropout_ratio=rand_input_dropout, 
                              hidden_dropout_ratios=rand_dropout,
                              shuffle_training_data = TRUE)
  
  models <- c(models, dlmodel)
  
  tmp = length(dlmodel@model$scoring_history$validation_classification_error)
  tmp_err <- dlmodel@model$scoring_history$validation_classification_error[tmp]
  if (tmp_err < tmp_min_err) {
    tmp_min_err = tmp_err
    tmp_best_model = dlmodel
    tmp_i = i
    tmp_act = rand_activation
    tmp_hidden = rand_hidden
  }
  sprintf("Accuracy %f; Max Accuracy %f",1-tmp_err,1-tmp_min_err)
  sprintf("Best params: %i | %s | %s ", tmp_i, tmp_act, paste(tmp_hidden,collapse=" "))
  if ((1-tmp_err) >= 0.90){
    break
  }
}

best_model <- models[[1]]
best_err <- 10000
for (i in 1:length(models)) {
  tmp = length(models[[i]]@model$scoring_history$validation_classification_error)
  err <- models[[i]]@model$scoring_history$validation_classification_error[tmp]
  if (err < best_err) {
    best_err <- err
    best_model <- models[[i]]
  }
}
1-best_err
#tmp = length(best_model@model$scoring_history$validation_classification_error)
#best_model@model
#$scoring_history$validation_classification_error[tmp]

#h2o.shutdown(FALSE)