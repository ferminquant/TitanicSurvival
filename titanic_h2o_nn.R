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

# for manual testing
nn = h2o.deeplearning(x = 2:8, y = 1, training_frame = hdata,#_train,
                      #validation_frame = hdata_test,
                      #nfolds = 10,
                      hidden = c(10),
                      standardize = TRUE,
                      activation = 'Tanh',
                      epochs = 100,
                      seed = 1320,
                      shuffle_training_data = TRUE,
                      variable_importances = TRUE)
#summary(nn)

#for hyperparameter optimization
models <- c()
tmp_min_err = 10000
i = 0

#Array to store distribution of model accuracy during hyperparameter optimization
acc_dist = NULL
for (i in 1:100) {
  acc_dist[i] = 0
}

i = 0
for (i in 1:10) {
#while (TRUE) {
  i = i + 1
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
  
  #models <- c(models, dlmodel)
  
  tmp = length(dlmodel@model$scoring_history$validation_classification_error)
  tmp_err <- dlmodel@model$scoring_history$validation_classification_error[tmp]
  
  j = round((1-tmp_err)*100,0)
  acc_dist[j] = acc_dist[j]+1
  
  if (tmp_err < tmp_min_err) {
    tmp_min_err = tmp_err
    tmp_best_model = dlmodel
    tmp_i = i
    tmp_act = rand_activation
    tmp_hidden = rand_hidden
  }
  print(sprintf("Accuracy %f; Max Accuracy %f",1-tmp_err,1-tmp_min_err))
  print(sprintf("Best params: %i | %s | %s ", tmp_i, tmp_act, paste(tmp_hidden,collapse=" ")))
  if ((1-tmp_err) >= 0.90){
    break
  }
}

best_err = tmp_min_err
best_model = tmp_best_model

# Save, Load, and Continue Training
#h2o.saveModel(best_model, path=getwd(), force=T)
#dlmodel_loaded <- h2o.loadModel(paste(getwd(),"DeepLearning_model_R_1482117924172_6685",sep="/"))
# dlmodel_continued_again <- h2o.deeplearning(x=2:8, y=1, 
#                                             training_frame = hdata_train, 
#                                             validation_frame = hdata_test, 
#                                             hidden=dlmodel_loaded@parameters$hidden,
#                                             activation=dlmodel_loaded@parameters$activation,
#                                             checkpoint = dlmodel_loaded@model_id, 
#                                             l1=dlmodel_loaded@parameters$l1,
#                                             l2=dlmodel_loaded@parameters$l2,
#                                             epochs=100)

#h2o.shutdown(FALSE)