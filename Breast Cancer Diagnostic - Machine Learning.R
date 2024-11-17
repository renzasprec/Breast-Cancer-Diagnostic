if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(matrixStats)
library(caret)
library(dslabs)
library(knitr)
library(doParallel)
options(digits = 3)
data(brca)

# for parallel computation
nc <- detectCores() - 1
cl <- makeCluster(nc)
registerDoParallel(cl)

# number of observations in the dataset
length(brca$y)

# number of features
ncol(brca$x)

# list of features
kable(colnames(brca$x), col.names = "features")

# models to be used for the breast cancer diagnostic algorithm
models<- c("kmeans","glm","lda","qda","loess", "knn","knn with pca", "rf", "ensemble")
kable(models,col.names = "models")

#### Preprocessing ####
# pre-processing of data by centering and scaling
pp<- preProcess(x = brca$x,method = c("center","scale"))
x_scaled<- predict(object = pp,newdata = brca$x)

#### Data Preparation ####
# partition data into a train and test set
set.seed(1996)
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- x_scaled[test_index,]
test_y <- brca$y[test_index]
train_x <- x_scaled[-test_index,]
train_y <- brca$y[-test_index]

# obtain validation set from the train set. The validation set would be used to calculate the final accuracy of the models
set.seed(1996)
validation_index <- createDataPartition(train_y, times = 1, p = 0.2, list = FALSE)
validation_x<- train_x[validation_index,]
validation_y<- train_y[validation_index]
train_x<- train_x[-validation_index,]
train_y <- train_y[-validation_index]

#### Data Exploration ####
# boxplot of each feature for each outcome
train_x |>
  data.frame() |>
  mutate(y=train_y) |>
  pivot_longer(-y,names_to = "feature",values_to = "value") |>
  ggplot() +
  geom_boxplot(aes(feature,value,fill=factor(y))) +
  theme(axis.text.x = element_text(angle = 90,hjust = 1))


#### Model Training ####

##### Training a k-means model #####

# define a function which takes two arguments - a matrix of observations x and a k-means object k - and assigns each row of x to a cluster from k.

predict_kmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}

# Create 25 bootstrap samples
set.seed(1996)
# a single vector of bootstrap indexes
bootstrap_indexes<- createResample(y = train_y,times = 25,list = FALSE) |> as.vector()

bootstrap_x<- train_x[bootstrap_indexes,]

# Use k-means to extract 2 centers from the 25 bootstrap samples. The centers correspond to the 2 outcomes
set.seed(1996)
k<- kmeans(bootstrap_x,centers = 2)

# predict using the predict_kmeans(function)
y_hat_kmeans<- predict_kmeans(x = test_x,k = k)
y_hat_kmeans<- ifelse(y_hat_kmeans==1,                       "M","B") |>
  factor(levels=levels(test_y))
  
# accuracy of k-means model
accuracy_kmeans<- confusionMatrix(data = y_hat_kmeans,reference = test_y)$overall["Accuracy"]

##### Logistic regression model #####
# Training of a logistic regression model using all predictors
set.seed(1996)
fit_glm<- train(x = train_x,y = train_y,method = "glm")

# prediction with glm model
y_hat_glm<- predict(object = fit_glm,newdata = test_x) |>
  factor(levels=levels(test_y))

# accuracy of glm
accuracy_glm<- confusionMatrix(data = y_hat_glm,reference = test_y)$overall["Accuracy"]

##### LDA and QDA models #####
# Train of LDA and QDA models
set.seed(1996)
fit_lda<- train(x = train_x,y = train_y,method = "lda") 
fit_qda<- train(x = train_x,y = train_y,method = "qda")

y_hat_lda<- predict(object = fit_lda,newdata = test_x) |>
  factor(levels=levels(test_y))
y_hat_qda<- predict(object = fit_qda,newdata = test_x) |>
  factor(levels=levels(test_y))

# accuracy of lda 
accuracy_lda<- confusionMatrix(data = y_hat_lda,reference = test_y)$overall["Accuracy"]

# accuracy of qda
accuracy_qda<- confusionMatrix(data = y_hat_qda,reference = test_y)$overall["Accuracy"]

##### Loess Model #####

# Training of gamLoess with default tuning parameters
set.seed(1996)
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
library(gam)

fit_loess<- train(x = train_x,y = train_y,method = "gamLoess")

# prediction with loess model
y_hat_loess<- predict(object = fit_loess,newdata=test_x) |>
  factor(levels=levels(test_y))
  
# Best tuning parameters
fit_loess$bestTune

# accuracy of gamLoess
accuracy_loess<- confusionMatrix(data = y_hat_loess,reference = test_y)$overall["Accuracy"]

##### K- nearest Nieghbors #####

# Training of a knn model
set.seed(1996)
fit_knn<- train(x = train_x,y = train_y,method = "knn",tuneGrid=data.frame(k=seq(3,21,2)))

# prediction with knn model
y_hat_knn<- predict(object = fit_knn,newdata=test_x) |>
  factor(levels=levels(test_y))

# Best tuning parameters
fit_knn$bestTune

# accuracy of knn
accuracy_knn<- confusionMatrix(data = y_hat_knn,reference = test_y)$overall["Accuracy"]


##### K-nearest Neighbors with PCA #####
# Training of a knn model with pca
pca<- prcomp(x = train_x)

# Obtain the minimum number of principal components required to explain at least 90% of the variance
p<- first(which((cumsum(pca$sdev^2)/sum(pca$sdev^2))>0.9))

# using the number of PCAs obtained, predict using a knn model
set.seed(1996)
fit_knn_pca<- train(x = train_x,y = train_y,method = "knn",tuneGrid=data.frame(k=seq(3,21,2)), preProcess = "pca",trControl = trainControl(preProcOptions = list(pcaComp=p)))

# prediction with knn model
y_hat_knn_pca<- predict(object = fit_knn_pca,newdata = test_x) |>
  factor(levels=levels(test_y))

# accuracy of knn model with pca
accuracy_knn_pca<- confusionMatrix(data = y_hat_knn_pca,reference = test_y)$overall["Accuracy"]

##### Random Forest #####
# Training of a random forest model
set.seed(1996)
fit_rf<- train(x = train_x,y = train_y,method = "rf",tuneGrid=data.frame(mtry=c(3,5,7,9)),importance=TRUE)

# Best tuning parameters
fit_rf$bestTune

# prediction with random forest model
y_hat_rf<- predict(object = fit_rf,newdata = test_x) |>
  factor(levels=levels(test_y))

# accuracy of random forest
accuracy_rf<- confusionMatrix(data = y_hat_rf,reference = test_y)$overall["Accuracy"]  
accuracy_rf

# 5 Most important variables
varImp(object = fit_rf)$importance |>
  slice_max(order_by = B,n = 5) 

# least important variable
varImp(object = fit_rf)$importance |>
  slice_min(order_by = B,n = 1) 


##### Ensemble model #####
# Create an ensemble model
ensemble<- data.frame(y_hat_kmeans,y_hat_glm,y_hat_lda,y_hat_qda,y_hat_loess,y_hat_knn,y_hat_knn_pca, y_hat_rf) |> 
  as.matrix()

# prediction with ensemble
y_hat_ensemble<- apply(X = ensemble,MARGIN = 1,FUN = function(x){
  ifelse(mean(x=="B")>0.5,"B","M") |>
    factor(levels=levels(test_y))
})

# accuracy of ensemble
accuracy_ensemble<- confusionMatrix(data = y_hat_ensemble,reference = test_y)$overall["Accuracy"]

# summary of models with corresponding accuracies
model_summary<- data.frame(model=c("kmeans","glm","lda","qda","knn","knn with pca", "rf","ensemble"), accuracy=c(accuracy_kmeans,accuracy_glm,accuracy_lda,accuracy_qda,accuracy_knn,accuracy_knn_pca, accuracy_rf,accuracy_ensemble))

kable(model_summary)

# model with the best accuracy
model_summary$model[which.max(model_summary$accuracy)]

# Use the ensemble model to evaluate the validation set

# k-means prediction
y_hat_kmeans_v<- predict_kmeans(x = validation_x,k = k)
y_hat_kmeans_v<- ifelse(y_hat_kmeans_v==1,                       "M","B") |>
  factor(levels=levels(test_y))

# glm prediction
y_hat_glm_v<- predict(object = fit_glm,newdata = validation_x) |>
  factor(levels=levels(test_y))

# qda and lda prediction
y_hat_lda_v<- predict(object = fit_lda,newdata = validation_x) |>
  factor(levels=levels(test_y))
y_hat_qda_v<- predict(object = fit_qda,newdata = validation_x) |>
  factor(levels=levels(test_y))

# loess prediction
y_hat_loess_v<- predict(object = fit_loess,newdata=validation_x) |>
  factor(levels=levels(test_y))

# knn prediction
y_hat_knn_v<- predict(object = fit_knn,newdata=validation_x) |>
  factor(levels=levels(test_y))

# knn with pca prediction
y_hat_knn_pca_v<- predict(object = fit_knn_pca,newdata = validation_x) |>
  factor(levels=levels(test_y))

# rf prediction
y_hat_rf_v<- predict(object = fit_rf,newdata = validation_x) |>
  factor(levels=levels(test_y))

# ensemble prediction
ensemble_v<- data.frame(y_hat_kmeans_v,y_hat_glm_v,y_hat_lda_v,y_hat_qda_v,y_hat_loess_v,y_hat_knn_v,y_hat_knn_pca_v, y_hat_rf_v) |> 
  as.matrix()
y_hat_ensemble_v<- apply(X = ensemble_v,MARGIN = 1,FUN = function(x){
  ifelse(mean(x=="B")>0.5,"B","M") |>
    factor(levels=levels(test_y))
})

# final accuracy on the validation set
accuracy_validation<- confusionMatrix(data = y_hat_ensemble_v, reference = validation_y)$overall["Accuracy"]

# stop parallelization
stopCluster(cl)
stopImplicitCluster()