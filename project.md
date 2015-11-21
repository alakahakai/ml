# Practical Machine Learning
Ray Qiu  
November 21st, 2015  

#### Project Overview

The goal of the project is to predict the manner in which people did the exercise. This is the "classe" variable in the training set. We can use any of the other variables to predict with. We will create a report describing how we built our model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices. We will also use the prediction model to predict 20 different test cases. 

#### Load libraries

```r
library(dplyr)
library(ggplot2)
library(caret)
library(rattle)
library(randomForest)
```
#### Data Processing

##### Read data in

```r
training <- read.csv("pml-training.csv", na.strings = c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv", na.strings = c("NA","#DIV/0!",""))
```

##### Quickly explore the data sets (Not evaluated in the report to save space)

```r
summary(training)
str(training)
head(training)
summary(testing)
str(testing)
head(testing)
```

##### Clean data

###### Remove columns that are simply an index, timestamp or username.

```r
training <- training[-c(1:6)]
testing <- testing[-c(1:6)]
```

###### Remove Near Zero-Variance Predictors

```r
nzv <- nearZeroVar(training)
training <- training[, -nzv]
nzv <- nearZeroVar(testing)
testing <- testing[,-nzv]
```

###### Find the columns that have more than 80% NA value.

```r
# Define a function to check the value
checkValue <- function(x) {
  if (is.na(x)) 1 else 0
}
# Return a vector for the total number of NA like values for each column
ret <- apply(training, 2, function(x) sum(sapply(x, checkValue)))
# Get the names of the columns that have more than 80% NA like values
n <- names(ret[ret > dim(training)[1] * 0.80])
```

######  Remove those columns from the training and testing data sets

```r
training <- select(training, -one_of(n))
testing <- select(testing, -one_of(n))
```

##### Proceed with further processing.  Set a fixed seed.

```r
set.seed(33833)
```

##### Partition training data set further into train1 and test1

```r
trainIndex <- createDataPartition(training$classe, p = 0.6, list = FALSE)
train1 <- training[trainIndex,]
test1 <- training[-trainIndex,]
dim(train1)
```

```
## [1] 11776    54
```

```r
dim(test1)
```

```
## [1] 7846   54
```

##### Train using random forest

```r
fit <- train(classe ~ ., data = train1, method = "rf", 
             preProcess = c("center", "scale"),
             trControl = trainControl(method="cv", number = 5),
             prox = TRUE, allowParallel = TRUE)
fit
```

```
## Random Forest 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (53), scaled (53) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9420, 9421, 9421, 9420, 9422 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD   
##    2    0.9908292  0.9883983  0.0024853279  0.003144827
##   27    0.9952446  0.9939851  0.0008163825  0.001032932
##   53    0.9928670  0.9909776  0.0021926747  0.002773463
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
plot(fit$finalModel, main = "Random Forest Final Model")
```

![](project_files/figure-html/unnamed-chunk-10-1.png) 

##### Predict using the test1 data set and check accuracy

```r
trainPredictions <- predict(fit, newdata = test1)
confusionMatrix(trainPredictions, test1$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    1    0    0    0
##          B    1 1517    0    0    0
##          C    0    0 1368    4    0
##          D    0    0    0 1282    8
##          E    0    0    0    0 1434
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9982        
##                  95% CI : (0.997, 0.999)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9977        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9993   1.0000   0.9969   0.9945
## Specificity            0.9998   0.9998   0.9994   0.9988   1.0000
## Pos Pred Value         0.9996   0.9993   0.9971   0.9938   1.0000
## Neg Pred Value         0.9998   0.9998   1.0000   0.9994   0.9988
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1933   0.1744   0.1634   0.1828
## Detection Prevalence   0.2845   0.1935   0.1749   0.1644   0.1828
## Balanced Accuracy      0.9997   0.9996   0.9997   0.9978   0.9972
```

```r
table(trainPredictions, test1$classe)
```

```
##                 
## trainPredictions    A    B    C    D    E
##                A 2231    1    0    0    0
##                B    1 1517    0    0    0
##                C    0    0 1368    4    0
##                D    0    0    0 1282    8
##                E    0    0    0    0 1434
```

##### Predict the 20 test cases that are provided 

```r
predictions <- predict(fit, newdata = testing)
predictions
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

#### Write answers to file

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(predictions)
```
