ae = h2o.deeplearning(x = 2:8, 
                      training_frame = hdata_train,
                      autoencoder = TRUE,
                      reproducible = T,
                      hidden = c(6,5,6), epochs = 50)


anomaly = h2o.anomaly(ae, hdata_train, per_feature=FALSE)
head(anomaly)
err <- as.data.frame(anomaly)
plot(sort(err$Reconstruction.MSE), main='Reconstruction Error')

hdata_normal  = hdata_train[anomaly$Reconstruction.MSE < 0.10,]
hdata_anomaly = hdata_train[anomaly$Reconstruction.MSE >= 0.10,]
NROW(hdata_normal)
NROW(hdata_anomaly)

RF_normal = h2o.randomForest(x=2:8,y=1,training_frame=hdata_train,
                      validation_frame = hdata_test,
                      nfolds=10,
                      ntrees=44,
                      max_depth=8,
                      mtries=7)

RF_anomaly = h2o.randomForest(x=2:8,y=1,training_frame=hdata_anomaly,
                             nfolds=10,
                             ntrees=44,
                             max_depth=8,
                             mtries=7)
