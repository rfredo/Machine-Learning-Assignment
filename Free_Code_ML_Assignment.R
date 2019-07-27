
#download data

library(caret)
library(corrplot)
library(rattle)
library(forecast)
library(rpart)
library(parallel)
library(doParallel)
library(xtable)

url1 <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
training <- basename(url1);
if (!file.exists(training)) {
    download.file(url1, training, method='curl')
}

url2 <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
testing <- basename(url2);
if (!file.exists(testing)) {
    download.file(url2, testing, method='curl')
}

#read data

pmlTraining <- read.csv(training, na.strings = c("", "NA"))
pmlTesting <- read.csv(testing, na.strings = c("", "NA"))

#create validation set
inTrain <- createDataPartition(y = pmlTraining$classe, p = .7, list = F)
training <- pmlTraining[inTrain,]
testing <- pmlTraining[-inTrain,]
dim(training); dim(testing)

#matNA <- matrix(nrow = 1, ncol = dim(training)[2], dimnames = list("%NA", names(training)))
#matNA[1,] <- as.numeric(matNA[1,])
#for (i in 1:dim(training)[2]) {
#    #matNA[1,i] <- names(training)[i]
#    matNA[1,i] <- mean(is.na(training[,i]))
#}
#matNA

#Establish the NAs % for every columns for feature selection
pmlNA <- sapply(training, FUN = function(x) mean(is.na(x)))
training <- training[,pmlNA<0.2]
testing <- testing[,pmlNA<0.2]

#eliminating average observations
training <- training[training$new_window == "no",]
testing <-testing[testing$new_window == "no",]

#remove feature with zero variance
nzv <- nearZeroVar(training)
training <- training[, -nzv]
testing <- testing[, -nzv]

#removing non-numeric first 5 columns

training <- training[,-c(1:5)]
testing <- testing[,-c(1:5)]

#featurePlot(x = training[,-59], training$classe)

corM <- cor(training[,-54])
corrplot(corM, order = "FPC", method = "color", type = "lower", tl.cex = .7)

#Prediction ML models
#cross validation

set.seed(323)
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

data_ctrl <- trainControl(method="cv", number = 5, allowParallel = T)

#predicting with decision trees

set.seed(323)
modFittree <- train(classe ~ ., method = "rpart", data = training, trControl = data_ctrl)
print(modFittree$finalModel)
fancyRpartPlot(modFittree$finalModel)
prpart <- predict(modFittree, testing)
confrpart <- confusionMatrix(prpart, testing$classe)
Accurtree <- conftree$overall[1]

#predicting with random forest

set.seed(323)
modFitrf <- train(classe ~ ., method = "rf", data = training, trControl = data_ctrl)
print(modFitrf$finalModel)
prf <- predict(modFitrf, testing)
confrf <- confusionMatrix(prf, testing$classe)
Accurrf <- confrf$overall[1]



#predicting eith GBM

set.seed(323)
modFitgbm <- train(classe ~ ., method = "gbm", data = training, verbose = F, trControl = data_ctrl)
print(modFitgbm$finalModel)
pgbm <- predict(modFitgbm, testing)
confgbm <- confusionMatrix(pgbm, testing$classe)
Accurgbm <- confgbm$overall[1]

#predicting eith LDA

set.seed(323)
modFitlda <- train(classe ~ ., method = "lda", data = training, trControl = data_ctrl)
print(modFitlda$finalModel)
plda <- predict(modFitlda, testing)
conflda <- confusionMatrix(plda, testing$classe)
Accurlda <- conflda$overall[1]

stopCluster(cluster)
registerDoSEQ()

table(Accurtree, Accurrf, Accurgbm, Accurlda)