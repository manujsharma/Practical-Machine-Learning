<h1> Practical Machine Learning Assignment </h1>

<h4> Analyze activity data to predict how well to predict how well enthusiasts are exercizing. </h4>

<h3> Data </h3>

<h4>
Training data:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

Testing data:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

</h4>

<h4> Load Relevant Libraries <h4>


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.3
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.3
```

```r
library(rpart)
```

```
## Warning: package 'rpart' was built under R version 3.2.3
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.2.3
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.2.3
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```


<h4> Load data </h4>


```r
set.seed(12345)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training_data <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
test_data <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

sum(training_data =='#DIV/0', na.rm=TRUE)
```

```
## [1] 0
```

```r
sum(test_data =='#DIV/0', na.rm=TRUE)
```

```
## [1] 0
```

```r
sum(training_data =='', na.rm=TRUE)
```

```
## [1] 0
```

```r
sum(test_data =='', na.rm=TRUE)
```

```
## [1] 0
```

<h4> Split the Training Dataset into Test and Training Data Sets</h4>


```r
Training_percent <- createDataPartition(training_data$classe, p=0.6, list=FALSE)

Training_dataset <- training_data[Training_percent, ]

Testing_dataset <- training_data[-Training_percent, ]

dim(Training_dataset); dim(Testing_dataset)
```

```
## [1] 11776   160
```

```
## [1] 7846  160
```

<h4> Remove Covariates with a) No Variance and b) Too many NAs </h4>


```r
No_Variance <- nearZeroVar(Training_dataset, saveMetrics=TRUE)

Training_dataset <- Training_dataset[,No_Variance$nzv==FALSE]

No_Variance<- nearZeroVar(Testing_dataset,saveMetrics=TRUE)

Testing_dataset<- Testing_dataset[,No_Variance$nzv==FALSE]

#Remove the first column of the Training data set as it is just the ID

Training_dataset <- Training_dataset[c(-1)]
```


<h4> Clean variables with more than 60% NA </h4>

```r
columns_To_Remove<-vector()

Count<-0

for(i in 1:ncol(Training_dataset)){
  if((sum(is.na(Training_dataset[,i]))/nrow(Training_dataset))>=0.60){
    Count<-Count+1
    columns_To_Remove[Count]<-i
  }
}
Training_dataset<-Training_dataset[,-columns_To_Remove]
dim(Training_dataset)
```

```
## [1] 11776    58
```

<h4> Transform the data sets so that they only contain Covariates that are present in the Training_dataset </h4>


```r
Cols1 <- colnames(Training_dataset)

# Remove the classe column that needs to be predicted
Cols2 <- colnames(Training_dataset[, -58])

#Update the Testing Dataset to have only the Covariates that are present in the Training dataset
Testing_dataset <- Testing_dataset[Cols1]       

#Update the Testing Dataset to have only the Covariates that are present in the Training dataset
test_data <- test_data[Cols2]           

dim(Testing_dataset)
```

```
## [1] 7846   58
```

```r
dim(test_data)
```

```
## [1] 20 57
```

<h4> Coerce the data into the same type to enable Machine Learning Algorithms to work </h4>


```r
for (i in 1:length(test_data) ) {
    for(j in 1:length(Training_dataset)) {
        if( length( grep(names(Training_dataset[i]), names(test_data)[j]) ) == 1)  {
            class(test_data[j]) <- class(Training_dataset[i])
        }   
    }   
}

# To get the same class between test_data and Training_dataset
test_data <- rbind(Training_dataset[2, -58] , test_data)
test_data <- test_data[-1,]
```

<h2> Prediction with Decision Trees </h2>


```r
set.seed(12345)

model_Fit_Dec_Tree <- rpart(classe ~ ., data=Training_dataset, method="class")

predictions_Dec_Tree <- predict(model_Fit_Dec_Tree, Testing_dataset, type = "class")

Confusion_Matrix_Dec_Tree <- confusionMatrix(predictions_Dec_Tree, Testing_dataset$classe)

Confusion_Matrix_Dec_Tree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2150   60    7    1    0
##          B   61 1260   69   64    0
##          C   21  188 1269  143    4
##          D    0   10   14  857   78
##          E    0    0    9  221 1360
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8789          
##                  95% CI : (0.8715, 0.8861)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8468          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9633   0.8300   0.9276   0.6664   0.9431
## Specificity            0.9879   0.9693   0.9450   0.9845   0.9641
## Pos Pred Value         0.9693   0.8666   0.7809   0.8936   0.8553
## Neg Pred Value         0.9854   0.9596   0.9841   0.9377   0.9869
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2740   0.1606   0.1617   0.1092   0.1733
## Detection Prevalence   0.2827   0.1853   0.2071   0.1222   0.2027
## Balanced Accuracy      0.9756   0.8997   0.9363   0.8254   0.9536
```

<h2> Prediction with Random Forests </h2>


```r
set.seed(12345)

model_Fit_Ran_For <- randomForest(classe ~ ., data=Training_dataset)

predictions_Ran_For <- predict(model_Fit_Ran_For, Testing_dataset, type = "class")

Confusion_Matrix_Dec_Tree <- confusionMatrix(predictions_Ran_For, Testing_dataset$classe)

Confusion_Matrix_Dec_Tree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    2    0    0    0
##          B    1 1516    0    0    0
##          C    0    0 1367    3    0
##          D    0    0    1 1282    1
##          E    0    0    0    1 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9989          
##                  95% CI : (0.9978, 0.9995)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9985          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9987   0.9993   0.9969   0.9993
## Specificity            0.9996   0.9998   0.9995   0.9997   0.9998
## Pos Pred Value         0.9991   0.9993   0.9978   0.9984   0.9993
## Neg Pred Value         0.9998   0.9997   0.9998   0.9994   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1932   0.1742   0.1634   0.1837
## Detection Prevalence   0.2846   0.1933   0.1746   0.1637   0.1838
## Balanced Accuracy      0.9996   0.9993   0.9994   0.9983   0.9996
```
<h2> Prediction with Generalized Boosted Regression </h2>


```r
set.seed(12345)

fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1)

Model_Fit_GBM <- train(classe ~ ., data=Training_dataset, method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE)
```

```
## Loading required package: plyr
```

```r
Final_Model_GBM <- Model_Fit_GBM$finalModel

predictions_GBM <- predict(Model_Fit_GBM, newdata=Testing_dataset)

Confusion_Matrix_GBM <- confusionMatrix(predictions_GBM, Testing_dataset$classe)

Confusion_Matrix_GBM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    3    0    0    0
##          B    1 1512    1    0    0
##          C    0    2 1361    2    0
##          D    0    1    6 1275    0
##          E    0    0    0    9 1442
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9968          
##                  95% CI : (0.9953, 0.9979)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.996           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9960   0.9949   0.9914   1.0000
## Specificity            0.9995   0.9997   0.9994   0.9989   0.9986
## Pos Pred Value         0.9987   0.9987   0.9971   0.9945   0.9938
## Neg Pred Value         0.9998   0.9991   0.9989   0.9983   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1927   0.1735   0.1625   0.1838
## Detection Prevalence   0.2847   0.1930   0.1740   0.1634   0.1849
## Balanced Accuracy      0.9995   0.9979   0.9971   0.9952   0.9993
```
<h2> Predicting Results on the Test Data </h2>

<h4>
Random Forests gave an Accuracy in the Testing dataset of 99.89%, which was more accurate than Decision Trees or GBM. The expected out-of-sample error is 100-99.89 = 0.11%.
</h4>

```r
Final_prediction <- predict(model_Fit_Ran_For, test_data, type = "class")

Final_prediction
```

```
##  2 31  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

