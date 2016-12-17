nn_test <- function(hidden_layers){
  #rm(list=ls(all=TRUE))
  library(neuralnet)
  
  set.seed(1320)
  data = read.csv("train.csv", header = TRUE)
  data$Age = ifelse(is.na(data$Age), -1, data$Age)
  
  accuracy = NULL
  error = NULL
  precision = NULL
  recall = NULL
  specificity = NULL
  fscore = NULL
  false_positive_rate = NULL
  
  data = model.matrix(~ Survived + Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data)
  maxs = apply(data, 2, max)
  mins = apply(data, 2, min)
  data = as.data.frame(scale(data, center = mins, scale = maxs - mins))
  
  data = data[sample(NROW(data)),]
  n_folds = 10
  folds = cut(seq(1,NROW(data)),breaks=n_folds,labels=FALSE)
  
  for(i in 1:n_folds){
    
    indexes = which(folds==i,arr.ind=TRUE)
    test_data = data[indexes, ]
    train_data = data[-indexes, ]
    
    model = neuralnet( 
      Survived ~ Pclass + Sexmale + Age + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS, 
      data=train_data, hidden=hidden_layers, threshold=0.05, stepmax=1e+07, lifesign='full',
      lifesign.step = 25000, rep=1, linear.output=FALSE
    )
    test_data$prediction = compute(model,test_data[,3:11])$net.result
    
    threshold = 0.5
    test_data$result = ifelse(test_data$Survived == 0 & test_data$prediction <  threshold, 'TN', 
                              ifelse(test_data$Survived == 0 & test_data$prediction >= threshold, 'FP', 
                                     ifelse(test_data$Survived == 1 & test_data$prediction >= threshold, 'TP', 'FN')))
    
    tmp = table(test_data$result)
    
    if(is.na(tmp['FN'])){FN = 0} else {FN = tmp['FN']}
    if(is.na(tmp['FP'])){FP = 0} else {FP = tmp['FP']}
    if(is.na(tmp['TN'])){TN = 0} else {TN = tmp['TN']}
    if(is.na(tmp['TP'])){TP = 0} else {TP = tmp['TP']}
    
    accuracy[i] = as.numeric((TN+TP)/sum(tmp))
    error[i] = as.numeric((FN+FP)/sum(tmp))
    precision[i] = as.numeric(TP/(TP+FP))
    recall[i] = as.numeric(TP/(TP+FN))
    specificity[i] = as.numeric(TN/(TN+FP))
    fscore[i] = 2*((precision[i]*recall[i])/(precision[i]+recall[i]))
    false_positive_rate[i] = as.numeric(FP/(TN+FP))
  }
  
  a = format(round(mean(accuracy)*100, 2), nsmall = 2)
  b = format(round(mean(error)*100, 2), nsmall = 2)
  c = format(round(mean(precision)*100, 2), nsmall = 2)
  d = format(round(mean(recall)*100, 2), nsmall = 2)
  e = format(round(mean(specificity)*100, 2), nsmall = 2)
  f = format(round(mean(fscore)*100, 2), nsmall = 2)
  g = format(round(mean(false_positive_rate)*100, 2), nsmall = 2)
  
  #print(sprintf("**%s%%** %s%% %s%% %s%% %s%% %s%% %s%%",a,b,c,d,e,f,g))
  retval = sprintf("**%s%%**   | %s%% | %s%%    | %s%% | %s%%      | %s%%   | %s%%  | ",a,b,c,d,e,f,g)
  retval
  
}

nn_test(c(20))