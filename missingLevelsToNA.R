missingLevelsToNA<-function(object,data){
  
  #Obtain factor predictors in the model and their levels ------------------
  
  factors<-(gsub("[-^0-9]|as.factor|\\(|\\)", "",names(unlist(object$xlevels))))
  factorLevels<-unname(unlist(object$xlevels))
  modelFactors<-as.data.frame(cbind(factors,factorLevels))
  
  
  #Select column names in your data that are factor predictors in your model -----
  
  predictors<-names(data[names(data) %in% factors])
  
  
  #For each factor predictor in your data if the level is not in the model set the value to NA --------------
  
  for (i in 1:length(predictors)){
    found<-data[,predictors[i]] %in% modelFactors[modelFactors$factors==predictors[i],]$factorLevels
    if (any(!found)) data[!found,predictors[i]]<-NA
  }
  
  data
  
}