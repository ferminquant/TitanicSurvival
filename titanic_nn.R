rm(list=ls(all=TRUE))
library(neuralnet)

data = read.csv("train.csv", header = TRUE)
data$Age = ifelse(is.na(data$Age), -1, data$Age)

data = data[sample(NROW(data)),]
folds = cut(seq(1,NROW(data)),breaks=4,labels=FALSE)
indexes = which(folds==1,arr.ind=TRUE)
test_data = data[indexes, ]
train_data = data[-indexes, ]

m = model.matrix(~ Survived + Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, train_data)
model = neuralnet( 
  Survived ~ Pclass + Sexmale + Age + SibSp + Parch + Fare + EmbarkedC + EmbarkedQ + EmbarkedS, 
  data=m, hidden=c(10), threshold=0.01, stepmax=1e+06, lifesign='full',
  lifesign.step = 25000, rep=1
)
n = model.matrix(~ Survived + Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, test_data)
test_data$prediction = compute(model,n[,3:11])$net.result

threshold = 0.5
test_data$result = ifelse(test_data$Survived == 0 & test_data$prediction <  threshold, 'TN', 
                    ifelse(test_data$Survived == 0 & test_data$prediction >= threshold, 'FP', 
                          ifelse(test_data$Survived == 1 & test_data$prediction >= threshold, 'TP', 'FN')))

tmp = table(test_data$result)

if(is.na(tmp['FN'])){FN = 0} else {FN = tmp['FN']}
if(is.na(tmp['FP'])){FP = 0} else {FP = tmp['FP']}
if(is.na(tmp['TN'])){TN = 0} else {TN = tmp['TN']}
if(is.na(tmp['TP'])){TP = 0} else {TP = tmp['TP']}

accuracy = as.numeric((TN+TP)/sum(tmp))
accuracy
