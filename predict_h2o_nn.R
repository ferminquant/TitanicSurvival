#Load and prepare
test = read.csv("test.csv", header = TRUE)
test$Age = ifelse(is.na(test$Age), -1, test$Age)
htest2 = as.h2o(test)
htest = htest2[-c(1,3,8,10)]

# dlmodel_loaded <- h2o.loadModel(paste(getwd(),"DeepLearning_model_R_1482117924172_6685",sep="/"))
# new_model <- h2o.deeplearning(x=2:8, y=1,
#                               training_frame = hdata,
#                               hidden=dlmodel_loaded@parameters$hidden,
#                               activation=dlmodel_loaded@parameters$activation,
#                               l1=dlmodel_loaded@parameters$l1,
#                               l2=dlmodel_loaded@parameters$l2,
#                               epochs=1000,
#                               classification_stop = -1)

htest2$Survived = as.numeric(h2o.predict(nn_aen,htest2)$predict)
h2o.exportFile(htest2[,c(1,12)],paste(path=getwd(),"h2o_nn_solutionNNAEN.csv",sep="/"),T)

# htest2$Survived1 = as.numeric(h2o.predict(RF_normal,htest2)$predict)
# htest2$Survived2 = as.numeric(h2o.predict(RF_anomaly,htest2)$predict)
# htest2$Survived = round((htest2$Survived1+htest2$Survived2)/2)
# h2o.exportFile(htest2[,c(1,14)],paste(path=getwd(),"h2o_nn_solutionAEcombined2.csv",sep="/"),T)
