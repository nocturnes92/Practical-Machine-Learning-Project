---
title: "Practical Machine Learning Project"
author: "Yigang"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---

# 1. Introduction 

## 1.1 Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

# 2. Data Processing

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## 2.1 Getting and Cleaning Data

### 2.1.1 Download and Read the Data


```r
# Globe environment setting
knitr::opts_chunk$set(warning=FALSE, message=FALSE, cache = TRUE)
```


```r
# Load libraries
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
```

### 2.1.2 Filter the Data

- Notice there are missing values in the data set, and variables provides information of observations but not for prediction, so we will neglect these data. 

```r
# See how many missing values in the data set
sum(complete.cases(RawTrainData))
```

```
## [1] 406
```

```r
# Remove all columns that contain missing values
TrainData <- RawTrainData[, colSums(is.na(RawTrainData)) == 0]
 
# Remove columns records users' names, timestamps and etc.
TrainData <- TrainData[, -c(1:7)]
```

## 2.2 Arranging Data

### 2.2.1 More Data Cleaning


```r
# remove variables have very little variation
remove <- nearZeroVar(TrainData)
TrainData <- TrainData[, -remove]
```


### 2.2.2 Slice the Data
- Split the training data into a training data set and a validation data set for the cross validation.


```r
# Set 30% of data from training data to be used for validation
# The "classe" variable in the training set is the manner in which users did the exercise
set.seed(1234)
partition <- createDataPartition(TrainData$classe, p=0.75, list=FALSE)
TrainData <- TrainData[partition, ]
TestData <- TrainData[-partition, ]
```

# 3. Modeling Data with Different Algorithms

We plan to use following methods to train and predict the data:

- Decision Tree
- Random Forests
- Gradient Boosted Method

## 3.1 Decision Tree Method

### 3.1.1 Training 


```r
# Train data through decision tree method
Model_CT <- rpart(classe ~ ., data = TrainData, method="class")

# Show tree by using fancyRpartPlot function
fancyRpartPlot(Model_CT)
```

![](Project_Report_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

### 3.1.2 Validating


```r
# Predict validation data with the model trained
Pred_CT <- predict(Model_CT, TestData, type = "class")

# See the accuracy of the prediction
confmat_CT <- confusionMatrix(Pred_CT, factor(TestData$classe))
confmat_CT
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 895 111  12  20  18
##          B  36 415  55  46  37
##          C  46 101 532  98  80
##          D  42  47  44 402  40
##          E  21  25  22  37 500
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7452          
##                  95% CI : (0.7308, 0.7593)
##     No Information Rate : 0.2825          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6779          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8606   0.5937   0.8000   0.6667   0.7407
## Specificity            0.9391   0.9417   0.8923   0.9438   0.9651
## Pos Pred Value         0.8475   0.7046   0.6208   0.6991   0.8264
## Neg Pred Value         0.9448   0.9082   0.9529   0.9353   0.9431
## Prevalence             0.2825   0.1898   0.1806   0.1638   0.1833
## Detection Rate         0.2431   0.1127   0.1445   0.1092   0.1358
## Detection Prevalence   0.2868   0.1600   0.2328   0.1562   0.1643
## Balanced Accuracy      0.8998   0.7677   0.8461   0.8052   0.8529
```

- Here we get a 0.75 prediction accuracy with the decision tree method, which is not that ideal, so we need to try other ways.

## 3.2 Random Forests Method

### 3.2.1 Training


```r
# Set k=3 in k-fold cross validation
set.seed(1334)
control_RF <- trainControl(method="cv", number=3, verboseIter=FALSE)

# Train data through random forests method
Model_RF <- train(classe ~ ., data = TrainData, method = "rf", trControl = control_RF)

# Check the statistical result 
Model_RF
```

```
## Random Forest 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 9811, 9813, 9812 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9896726  0.9869342
##   27    0.9905558  0.9880522
##   52    0.9809076  0.9758432
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

### 3.2.2 Validating


```r
# Predict validation data with the model trained 
Pred_RF <- predict(Model_RF, TestData)

# See the accuracy of the prediction
confmat_RF <- confusionMatrix(Pred_RF, factor(TestData$classe))
confmat_RF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1040    0    0    0    0
##          B    0  699    0    0    0
##          C    0    0  665    0    0
##          D    0    0    0  603    0
##          E    0    0    0    0  675
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.2825    
##     P-Value [Acc > NIR] : < 2.2e-16 
##                                     
##                   Kappa : 1         
##                                     
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2825   0.1898   0.1806   0.1638   0.1833
## Detection Rate         0.2825   0.1898   0.1806   0.1638   0.1833
## Detection Prevalence   0.2825   0.1898   0.1806   0.1638   0.1833
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

- Here we get incredible perfect prediction accuracy with the random forests method, so we want to have a deeper view of this model.


```r
# A plot of number of trees versus error of model
plot(Model_RF$finalModel)
```

![](Project_Report_files/figure-html/unnamed-chunk-10-1.png)<!-- -->

```r
# See the importance of each variables in this model
varImp(Model_RF)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt            100.000
## pitch_forearm         59.254
## yaw_belt              53.449
## pitch_belt            44.098
## roll_forearm          43.743
## magnet_dumbbell_z     43.417
## magnet_dumbbell_y     42.247
## accel_dumbbell_y      22.221
## accel_forearm_x       18.098
## roll_dumbbell         17.175
## magnet_belt_z         15.877
## magnet_dumbbell_x     15.631
## magnet_forearm_z      14.729
## accel_belt_z          14.115
## accel_dumbbell_z      13.393
## total_accel_dumbbell  13.213
## magnet_belt_y         12.793
## yaw_arm               10.630
## gyros_belt_z          10.605
## magnet_belt_x          9.778
```

## 3.3 Gradient Boosted Method

### 3.3.1 Training 


```r
# Set seed for reproduce
set.seed(1434)
control_GBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)

# Train data through gradient boosted method
Model_GBM <- train(classe ~ ., data = TrainData, method = "gbm",
                   trControl = control_GBM, verbose = FALSE)

# Check the statistical result 
Model_GBM
```

```
## Stochastic Gradient Boosting 
## 
## 14718 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 1 times) 
## Summary of sample sizes: 11774, 11775, 11773, 11775, 11775 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7533633  0.6874558
##   1                  100      0.8191337  0.7710841
##   1                  150      0.8521547  0.8129341
##   2                   50      0.8558908  0.8174003
##   2                  100      0.9063051  0.8814493
##   2                  150      0.9298817  0.9112816
##   3                   50      0.8957058  0.8679918
##   3                  100      0.9398011  0.9238241
##   3                  150      0.9612040  0.9509200
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150, interaction.depth =
##  3, shrinkage = 0.1 and n.minobsinnode = 10.
```

### 3.3.2 Validating


```r
# Predict validation data with the model trained 
Pred_GBM <- predict(Model_GBM, TestData)

# See the accuracy of the prediction
confmat_GBM <- confusionMatrix(Pred_GBM, factor(TestData$classe))
confmat_GBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1033   15    0    1    2
##          B    6  676   17    2    3
##          C    0    6  639   16    6
##          D    1    2    9  582   10
##          E    0    0    0    2  654
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9734          
##                  95% CI : (0.9677, 0.9783)
##     No Information Rate : 0.2825          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9663          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9933   0.9671   0.9609   0.9652   0.9689
## Specificity            0.9932   0.9906   0.9907   0.9929   0.9993
## Pos Pred Value         0.9829   0.9602   0.9580   0.9636   0.9970
## Neg Pred Value         0.9973   0.9923   0.9914   0.9932   0.9931
## Prevalence             0.2825   0.1898   0.1806   0.1638   0.1833
## Detection Rate         0.2806   0.1836   0.1735   0.1581   0.1776
## Detection Prevalence   0.2854   0.1912   0.1812   0.1640   0.1782
## Balanced Accuracy      0.9932   0.9789   0.9758   0.9790   0.9841
```

- Here we also get good prediction accuracy with the gradient boosted method, so again we want to explore this model.


```r
# A Boosting Iterations versus accuracy plot
plot(Model_GBM)
```

![](Project_Report_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

# 4. Conclusion

- Decision Trees Model is no a suitable model for this data set.
- Random Forests Model has the best prediction accuracy, so we will apply it to predict  ``` TestSet```. 
- Gradient Boosted Method also has a high prediction accuracy, which is just a little bit lower than Random Forests Model's, and it may be an alternate option in the future.

## 4.1 Apply Random Forests Model to the Test Set 


```r
TestPred <- predict(Model_RF, TestSet)
TestPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

