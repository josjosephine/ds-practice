# Lab: Classification Methods

#Naive Bayes classifier - Gender prediction

set.seed(4240)  #for reproducibility
gender = as.factor(c("M","M","F","F","F","M","M","M","M","F","F","F"))
height = c(1.70, 1.67, 1.60, 1.62, 1.54, 1.82, 1.75,
           1.7, 1.69, 1.6, 1.70, 1.70)
weight = c(70, 71, 60, 50, 55, 80, 68, 62, 69, 55, 66, 70)
shoes = as.factor(c("Few","Avg","Lot","Lot","Avg",
                    "Avg","Few","Lot","Lot","Avg","Avg","Lot"))
data.training = data.frame(gender, height, weight, shoes)
data.training

#Library for Naive Bayes Classifier
# install.packages("e1071")
library(e1071)

#laplace = regularization parameter
classifier.NB = naiveBayes(gender ~ .,data.training, laplace = 0)
classifier.NB

classifier.NB.regularized = naiveBayes(gender ~ .,data.training, laplace = 2)
classifier.NB.regularized

#How would you classify an individual who has height 1.70, weight - 65 and has a few number of shoes?
nb.prediction = predict(classifier.NB, data.frame(height=1.7, weight=65, shoes="Few"), type="raw")
nb.prediction

# regularized
nb.prediction = predict(classifier.NB.regularized, data.frame(height=1.7, weight=65, shoes="Few"), type="raw")
nb.prediction


### K nearest neighbours (Insurance - Caravan data)
# install.packages("ISLR2")
library(ISLR2)
# install.packages("class")
library(class)

dim(Caravan)
attach(Caravan)
summary(Purchase)
348 / 5822
###
standardized.X <- scale(Caravan[, -86])
var(Caravan[, 1])
var(Caravan[, 2])
var(standardized.X[, 1])
var(standardized.X[, 2])

### Split train-test
test <- 1:1000
train.X <- standardized.X[-test, ]
test.X <- standardized.X[test, ]
train.Y <- Purchase[-test]
test.Y <- Purchase[test]

set.seed(1)
knn.pred <- knn(train.X, test.X, train.Y, k = 1)
mean(test.Y != knn.pred)
mean(test.Y != "No")

###
table(knn.pred, test.Y)
9 / (68 + 9)
###
knn.pred <- knn(train.X, test.X, train.Y, k = 3)
table(knn.pred, test.Y)
5 / 26
knn.pred <- knn(train.X, test.X, train.Y, k = 5)
table(knn.pred, test.Y)
4 / 15

### Compare with Logistic Regression
glm.fits <- glm(Purchase ~ ., data = Caravan,
                family = binomial, subset = -test)
glm.probs <- predict(glm.fits, Caravan[test, ],
                     type = "response")
glm.pred <- rep("No", 1000)
glm.pred[glm.probs > .5] <- "Yes"
table(glm.pred, test.Y)
glm.pred <- rep("No", 1000)
glm.pred[glm.probs > .25] <- "Yes"
table(glm.pred, test.Y)
11 / (22 + 11)


### Naive Bayes Workhop - impact of webpage activity on market sales (Smarket)

library(e1071)
library(ISLR2)

Smarket=read.csv("webpage.csv")
attach(Smarket)

str(Smarket)
names(Smarket)
dim(Smarket)
summary(Smarket)


### split train test
train <- (Year < 2005)
Smarket.2005 <- Smarket[!train, ]

dim(Smarket.2005)
Direction.2005 <- Direction[!train] #Direction column onlly

### mean and SD for variable X1
mean(X1[train][Direction[train] == "Down"])
sd(X1[train][Direction[train] == "Down"])

### Fit Naive Bayes - X1 and X2
nb.fit <- naiveBayes(Direction ~ X1 + X2, data = Smarket,
                     subset = train)
nb.fit

### predict on testset
nb.class <- predict(nb.fit, Smarket.2005)
table(nb.class, Direction.2005)
### accuracy rate
mean(nb.class == Direction.2005) #0.59

### The predict() function can also generate estimates of the probability that each observation belongs to a particular class. %
nb.preds <- predict(nb.fit, Smarket.2005, type = "raw")
nb.preds[1:5, ]

### Fit Naive Bayes (X1-X5) - try it out! 
nb.fit <- naiveBayes(Direction ~ .-Year -Volume -Sales, data = Smarket,
                     subset = train)
nb.fit

### predict on testset
nb.class <- predict(nb.fit, Smarket.2005)
table(nb.class, Direction.2005)
### accuracy rate
mean(nb.class == Direction.2005) #0.579, worse than X1-X2

### The predict() function can also generate estimates of the probability that each observation belongs to a particular class. %
nb.preds <- predict(nb.fit, Smarket.2005, type = "raw")
nb.preds[1:5, ]

## KNearest Neighbours Workshop - impact of webpage activity on market sales (Smarket) 
library(class)
train.X <- cbind(X1, X2)[train, ]
test.X <- cbind(X1, X2)[!train, ]
train.Direction <- Direction[train]

set.seed(1)
knn.pred <- knn(train.X, test.X, train.Direction, k = 1)
table(knn.pred, Direction.2005)
# (83 + 43) / 252
mean(knn.pred == Direction.2005) #0.5
43/(43 + 58)
43/(43+68)

#try K=2 and K=3 class exercise - try it out! 
# knn.pred <- knn(train.X, test.X, train.Direction, k = 2)
knn.pred <- knn(train.X, test.X, train.Direction, k = 3)
table(knn.pred, Direction.2005)
mean(knn.pred == Direction.2005) #0.53 when k=2, 0.54 when k=3
48/(48+55)
48/(48+63)

#####################################################
# Lab: Support Vector Machines


## Support Vector Classifier

### Example 1 - Linear classifier
set.seed(1)
x <- matrix(rnorm(20 * 2), ncol = 2)
y <- c(rep(-1, 10), rep(1, 10)) #halve the matrix x
x[y == 1, ] <- x[y == 1, ] + 1 # move the datapoints further 
plot(x, col = (3 - y)) # col=color, can search for the col map

###create a data frame with the response coded as a factor.
dat <- data.frame(x = x, y = as.factor(y))
library(e1071)
svmfit <- svm(y ~ ., data = dat, kernel = "linear", 
              cost = 10, scale = FALSE)
plot(svmfit, dat)

### train using cost=10, seven support vectors
svmfit$index

summary(svmfit)
### Use a smaller value of cost parameter - larger number of support vectors obtained
svmfit <- svm(y ~ ., data = dat, kernel = "linear", 
              cost = 0.1, scale = FALSE)
plot(svmfit, dat)
dev.off()
svmfit$index # There are 16 support vectors
summary(svmfit)
### The e1071 library includes tune function to perform cross validation

### tune using cross-validation
set.seed(1)
tune.out <- tune(svm, y ~ ., data = dat, kernel = "linear", 
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

summary(tune.out)
### select optimal model
bestmod <- tune.out$best.model
summary(bestmod)

### generate testset
xtest <- matrix(rnorm(20 * 2), ncol = 2)
ytest <- sample(c(-1, 1), 20, rep = TRUE)
xtest[ytest == 1, ] <- xtest[ytest == 1, ] + 1
testdat <- data.frame(x = xtest, y = as.factor(ytest))

### predict on testset
ypred <- predict(bestmod, testdat)
table(predict = ypred, truth = testdat$y) # predicted vs actual
mean(ypred==testdat$y) #accuracy rate! 

### With cost =0.01, more observations misclassified 
svmfit <- svm(y ~ ., data = dat, kernel = "linear", 
              cost = .01, scale = FALSE)
ypred <- predict(svmfit, testdat)
table(predict = ypred, truth = testdat$y) # predicted vs actual
mean(ypred==testdat$y)

### NB test
#Library for Naive Bayes Classifier
# install.packages("e1071")
library(e1071)

#laplace = regularization parameter
classifier.NB = naiveBayes(y ~ .,dat, laplace = 0)
classifier.NB

#Prediction
nb.prediction = predict(classifier.NB, testdat)
table(nb.prediction, testdat$y)
mean(nb.prediction==testdat$y)

#### SVM Workshop Linear classifier Exercise 1
x[y == 1, ] <- x[y == 1, ] +0.5
plot(x, col = (y + 5) / 2, pch = 19)
dat <- data.frame(x = x, y = as.factor(y))

# try large c
svmfit <- svm(y ~ ., data = dat, kernel = "linear", 
              cost = 1e5)
summary(svmfit)
plot(svmfit, dat)
# try small c
svmfit <- svm(y ~ ., data = dat, kernel = "linear", cost = 1)
summary(svmfit)

# Step 2 - tune parameters (Try it out!)
set.seed(1)
tune.out <- tune(svm, y ~ ., data = dat, kernel = "linear", 
                 ranges = list(cost = c(0.1,1, 10, 1e2, 1e3, 1e4, 1e5)))

summary(tune.out)

### select optimal model
bestmod <- tune.out$best.model
summary(bestmod)

### generate testset
xtest <- matrix(rnorm(20 * 2), ncol = 2)
ytest <- sample(c(-1, 1), 20, rep = TRUE)
xtest[ytest == 1, ] <- xtest[ytest == 1, ] + 1
testdat <- data.frame(x = xtest, y = as.factor(ytest))

### Step 3 - Testset Prediction (Try it out!)
ypred <- predict(bestmod, testdat)
table(predict = ypred, truth = testdat$y) # predicted vs actual
mean(ypred==testdat$y) #accuracy rate!  #0.8 
# results changed might because: 1. set seed. 
# 2. small sample size causes different classification, may not exist for large size.

### With cost =0.1, more observations are misclassified 
svmfit <- svm(y ~ ., data = dat, kernel = "linear", 
              cost = .1, scale = FALSE)
ypred <- predict(svmfit, testdat)
table(predict = ypred, truth = testdat$y) # predicted vs actual
mean(ypred==testdat$y)  #0.85 manually might be slightly different from auto one.

### Compare with Naive Bayes (try it out!)
library(e1071)

#laplace = regularization parameter
classifier.NB = naiveBayes(y ~ .,dat, laplace = 0)
classifier.NB

#Prediction
nb.prediction = predict(classifier.NB, testdat)
table(nb.prediction, testdat$y)
mean(nb.prediction==testdat$y) #0.75


## Example 2 - non-linear

### generate training data
set.seed(1)
x <- matrix(rnorm(200 * 2), ncol = 2)
x[1:100, ] <- x[1:100, ] + 2
x[101:150, ] <- x[101:150, ] - 2
y <- c(rep(1, 150), rep(2, 50))
dat <- data.frame(x = x, y = as.factor(y))
plot(x, col = y)
# sample 100 out of 200 for trainset
train <- sample(200, 100)

# try small c
svmfit <- svm(y ~ ., data = dat[train, ], kernel = "radial",  
              gamma = 1, cost = 1)

plot(svmfit, dat[train, ])

summary(svmfit)

### try large c
svmfit <- svm(y ~ ., data = dat[train, ], kernel = "radial", 
              gamma = 1, cost = 1e5)
plot(svmfit, dat[train, ])
summary(svmfit)

### tune the model (non-linear)
set.seed(1)
tune.out <- tune(svm, y ~ ., data = dat[train, ], 
                 kernel = "radial", 
                 ranges = list(
                   cost = c(0.1, 1, 10, 100, 1000),
                   gamma = c(0.5, 1, 2, 3, 4)
                 )
)
summary(tune.out)
bestmod <- tune.out$best.model
summary(bestmod) # 30 support vectors

### Predict on testset
table(
  true = dat[-train, "y"], 
  pred = predict(
    tune.out$best.model, newdata = dat[-train, ])
)
  
ypred=predict(tune.out$best.model, newdata = dat[-train, ])
testdat=dat[-train,]

mean(ypred==testdat$y) #0.88

### section{Naive Bayes classifier to compare with SVM non-linear classifier}
classifier.NB = naiveBayes(y ~ .,dat, laplace = 0)
classifier.NB

#Prediction
nb.prediction = predict(classifier.NB, testdat)
table(nb.prediction, testdat$y)
mean(nb.prediction==testdat$y)  #0.86

### SVM Workshop Exercise 2: Application of SVMs to High Dimensional Data (large p)

###
# install.packages("ISLR2")
library(ISLR2)
library(e1071)
str(Khan)
names(Khan)
dim(Khan$xtrain)
dim(Khan$xtest)
length(Khan$ytrain)
length(Khan$ytest)

###
table(Khan$ytrain)
table(Khan$ytest)
###
dat <- data.frame(
  x = Khan$xtrain, 
  y = as.factor(Khan$ytrain)
)
str(dat)
### train using cost=10
out <- svm(y ~ ., data = dat, kernel = "linear", 
           cost = 10)
summary(out)
table(out$fitted, dat$y)

### predict on testset
testdat <- data.frame(
  x = Khan$xtest, 
  y = as.factor(Khan$ytest))
pred.te <- predict(out, newdata = testdat)
table(pred.te, testdat$y) # linear SVM predicts well
mean(pred.te==testdat$y) # 0.9
###

### section{Naive Bayes classifier to compare with SVM classifier}
classifier.NB = naiveBayes(y ~ .,dat, laplace = 0)
classifier.NB

#Prediction
nb.prediction = predict(classifier.NB, testdat)
table(nb.prediction, testdat$y)
mean(nb.prediction==testdat$y) #0.45

#### The following codes  may encounter running issues in R.  Try in Google Colab instead 

### tune model (linear) - run in google colab! 
set.seed(1)
tune.out <- tune(svm, y ~ ., data = dat, kernel = "linear", 
                 ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

summary(tune.out)
### select optimal model
bestmod <- tune.out$best.model
summary(bestmod)
### predict on testset
testdat <- data.frame(
  x = Khan$xtest, 
  y = as.factor(Khan$ytest))
pred.te <- predict(out, newdata = testdat)
table(pred.te, testdat$y) # linear SVM predicts well
mean(pred.te==testdat$y)
###

### tune the model (non-linear) - try it out in google colab! 
set.seed(1)
tune.out <- tune(svm, y ~ ., data = dat[train, ], 
                 kernel = "radial", 
                 ranges = list(
                   cost = c(0.1, 1, 10, 100, 1000),
                   gamma = c(0.5, 1, 2, 3, 4)
                 )
)
summary(tune.out)
bestmod <- tune.out$best.model
summary(bestmod)

### predict on testset
testdat <- data.frame(
  x = Khan$xtest, 
  y = as.factor(Khan$ytest))
pred.te <- predict(out, newdata = testdat)
table(pred.te, testdat$y) # linear SVM predicts well
mean(pred.te==testdat$y)
###

### results 
## non-linear:
# best parameter: cost = 0.1, gamma = 0.5
# number of support vectors: 63
# test accuracy: 0.9

## linear:
# best parameter: cost = 0.001
# number of support vectors:58
# test accuracy: 0.9

# NB:
# test accuracy: 0.45