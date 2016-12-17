rm(list=ls(all=TRUE))

set.seed(1320)
data = read.csv("train.csv", header = TRUE)
data$Age = ifelse(is.na(data$Age), -1, data$Age)
formula = Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked

final_accuracy = NULL
final_error = NULL
final_precision = NULL
final_recall = NULL
final_specificity = NULL
final_fscore = NULL
final_fp_rate = NULL

for(j in 1:100){
  print(sprintf("Start %i", j))
  accuracy = NULL
  error = NULL
  precision = NULL
  recall = NULL
  specificity = NULL
  fscore = NULL
  false_positive_rate = NULL
  
  #Randomly shuffle the data
  data = data[sample(NROW(data)),]
  
  #Create 10 equally size folds
  n_folds = 10
  folds = cut(seq(1,NROW(data)),breaks=n_folds,labels=FALSE)
  
  #Perform 10 fold cross validation
  for(i in 1:n_folds){
    #print(sprintf("Inner loop %i, %i", j, i))
    indexes = NULL
    test_data = NULL
    train_data = NULL
    indexes = which(folds==i,arr.ind=TRUE)
    test_data = data[indexes, ]
    train_data = data[-indexes, ]
    
    model = glm(formula, family = binomial(link='logit'), train_data)
    
    source("missingLevelsToNA.R")
    test_data = missingLevelsToNA(model,test_data)
    
    #glm predict
    test_data$prediction = predict(model, newdata=test_data, type='response')
    
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
    
    #of all predicted positives, how many were really positive
    precision[i] = as.numeric(TP/(TP+FP))
    
    #of all positives, how many were predicted as positive
    #also know as sensitivity or true positive rate
    recall[i] = as.numeric(TP/(TP+FN))
    
    #of all negatives, how many were predicted as negative
    specificity[i] = as.numeric(TN/(TN+FP))
    
    fscore[i] = 2*((precision[i]*recall[i])/(precision[i]+recall[i]))
    
    #of all negatives, how many were predicted as positive
    false_positive_rate[i] = as.numeric(FP/(TN+FP))
  }
  
  final_accuracy[j] = mean(accuracy)
  final_error[j] = mean(error)
  final_precision[j] = mean(precision)
  final_recall[j] = mean(recall)
  final_specificity[j] = mean(specificity)
  final_fscore[j] = mean(fscore)
  final_fp_rate[j] = mean(false_positive_rate)
  
}

c(mean(final_accuracy), sd(final_accuracy))
c(mean(final_error), sd(final_error))
c(mean(final_precision), sd(final_precision))
c(mean(final_recall), sd(final_recall))
c(mean(final_specificity), sd(final_specificity))
c(mean(final_fscore), sd(final_fscore))
c(mean(final_fp_rate), sd(final_fp_rate))
