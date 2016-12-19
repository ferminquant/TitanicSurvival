#Load and prepare
test = read.csv("test.csv", header = TRUE)
test$Age = ifelse(is.na(test$Age), -1, test$Age)
htest2 = as.h2o(test)
htest = htest2[-c(1,3,8,10)]

dlmodel_loaded <- h2o.loadModel(paste(getwd(),"DeepLearning_model_R_1482117924172_6685",sep="/"))
new_model <- h2o.deeplearning(x=2:8, y=1,
                              training_frame = hdata,
                              hidden=dlmodel_loaded@parameters$hidden,
                              activation=dlmodel_loaded@parameters$activation,
                              l1=dlmodel_loaded@parameters$l1,
                              l2=dlmodel_loaded@parameters$l2,
                              epochs=1000,
                              classification_stop = -1)

htest2$Survived = as.numeric(h2o.predict(RF,htest2)$predict)
h2o.exportFile(htest2[,c(1,12)],paste(path=getwd(),"h2o_nn_solutionRF.csv",sep="/"),T)

#hpo = 0.74163
#100x3 = 0.71292
#60 = 0.76555
#MD4 = 0.74641
#10 = 0.6910
#100x2 = 0.71770
#100 = 0.74641