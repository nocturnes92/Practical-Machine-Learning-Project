# Globe environment setting
knitr::opts_chunk$set(warning=FALSE, message=FALSE, cache = TRUE)
library(lattice)
library(ggplot2)
library(ggcorrplot)
library(caret)
library(rattle)
library(rpart)
library(rpart.plot)
library(corrplot)

# Import data sets
TrainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
RawTrainData <- read.csv(url(TrainUrl), header = TRUE)
TestSet <- read.csv(url(TestUrl), header = TRUE)

# See how many missing values in the data set
sum(complete.cases(RawTrainData))

# Remove all columns that contain missing values
TrainData <- RawTrainData[, colSums(is.na(RawTrainData)) == 0]

# Remove columns records users' names, timestamps and etc.
TrainData <- TrainData[, -c(1:7)]

# remove variables have very little variation
remove <- nearZeroVar(TrainData)
TrainData <- TrainData[, -remove]

# Set 30% of data from training data to be used for validation
# The "classe" variable in the training set is the manner in which users did the exercise
set.seed(1234)
partition <- createDataPartition(TrainData$classe, p=0.75, list=FALSE)
TrainData <- TrainData[partition, ]
TestData <- TrainData[-partition, ]

# Train data through classification tree method
Model_CT <- rpart(classe ~ ., data = TrainData, method="class")

# Show tree by using fancyRpartPlot function
fancyRpartPlot(Model_CT)

# Set k=3 in k-fold cross validation
set.seed(1334)
control_RF <- trainControl(method="cv", number=3, verboseIter=FALSE)

# Train data through random forests method
Model_RF <- train(classe ~ ., data = TrainData, method = "rf", trControl = control_RF)

# Check the statistical result 
Model_RF

# Predict validation data with the model trained 
Pred_RF <- predict(Model_RF, TestData)

# See the accuracy of the prediction
confmat_RF <- confusionMatrix(Pred_RF, factor(TestData$classe))
confmat_RF

# A plot of number of trees versus error of model
plot(Model_RF$finalModel)

# See the importance of each variables in this model
varImp(Model_RF)

# Set seed for reproduce
set.seed(1434)
control_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)

# Train data through random forests method
Model_GBM <- train(classe ~ ., data = TrainData, method = "gbm",
                   trControl = control_GBM, verbose = FALSE)

# Check the statistical result 
Model_GBM

# Predict validation data with the model trained 
Pred_GBM <- predict(Model_GBM, TestData)

# See the accuracy of the prediction
confmat_GBM <- confusionMatrix(Pred_GBM, factor(TestData$classe))
confmat_GBM

# A Boosting Iterations versus accuracy plot
plot(Model_GBM)

TestPred <- predict(Model_RF, TestSet)
TestPred
