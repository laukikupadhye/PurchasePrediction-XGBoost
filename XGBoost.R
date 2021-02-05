#---------------------------------------------- RF and GLM with crossvalidation-------------
library(caret)
library(randomForest)

data=read.csv('train.csv',header = T)
test=read.csv('test.csv',header = T)
#str(data)
data$Buy=as.factor(data$Buy)
data=data[,-1]
#options(warn=-1)
trainCont <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 5, 
                     verboseIter = FALSE,
                     sampling = "up")
set.seed(608)
#options(warn=1)
glm <- train(Buy ~.,
            data = data,
            method = "glm",
            preProcess = c("scale", "center"),
            trControl = trainCont)

p4_train=predict(glm,data)
confusionMatrix(p4_train,data$Buy) 
p4_test=predict(glm,test) 
submission <- data.frame("Id" = test$Id,"Predicted" =p4_test)
write.csv(submission,"submit.csv", row.names = FALSE)
table(submission$Predicted)

#---------------------------------------------- XGBoost ------------------------------------
library(dplyr)
library(xgboost)
library(caret)

train=read.csv('../input/classification-data-challenge/train.csv',header = T)
test=read.csv('../input/classification-data-challenge/test.csv',header = T)
test_modelData=test[,-1]
#str(train)
train=train[,-1]
train <- data.matrix(train, rownames.force = NA)

trainMat <- xgb.DMatrix(data = train[,-86], label = train[,86]) 

para <- list(booster = "gbtree", 
               objective = "binary:logistic", 
               eta=0.07, 
               gamma=0.01, 
               max_depth=6, 
               min_child_weight=1, 
               subsample=1, 
               colsample_bytree=1)
set.seed(608)
xgbtraincv <- xgb.cv( params = para, data = trainMat, 
                nrounds = 410, 
                nfold = 5, 
                showsd = T, 
                stratified = T, 
                early_stopping_rounds = 21, 
                maximize = F)

set.seed(608)
xgb_train <- xgb.train (params = para, data = trainMat, nrounds = 15)

test_modelData <- data.matrix(test_modelData, rownames.force = NA)
pred_test <-  predict(xgb_train, test_modelData)

submission <- data.frame("Id" = test$Id, "Predicted" =pred_test)
write.csv(submission,"./submission_xgb.csv", row.names = FALSE)