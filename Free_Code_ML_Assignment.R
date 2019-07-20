
#download data

library(caret)
library(corrplot)

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
inTrain <- createDataPartition(y = pmlTraining$classe, p = .7, list - F)
training <- pmlTraining[inTrain,]
validation <- pmlTraining[-inTrain,]
dim(training); dim(validation)

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
validation <- validation[,pmlNA<0.2]

training <- training[training$new_window == "no",]
validation <-validation[validation$new_window == "no",]

nzv <- nearZeroVar(training)
training <- training[, -nzv]
validation <- validation[, -nzv]

featurePlot(x = training[,-59], training$classe)

corM <- cor(training[,-c(1:5,59)])
corrplot.mixed(corM, order = "FPC", number.cx = .7)
