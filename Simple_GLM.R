# PA Workshop Diabetes exercise

pacman::p_load(tidyverse, caret, corrplot, caTools,knitr,car, ROCR,IRdisplay, e1071, earth, riv, woe,ROSE)

data =  read.csv('diabetes.csv')

# data overview
head(data,4)
str(data)
summary(data)

# from the correlation, we don't see signigicant collinearity between independent variables
# from the last column, the Glucose has the highest relationship with the Outcome.
pairs(~Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age,data=data, main="Scatterplot Matrix")
cor(data)

# set the variables to factors (categorical data)
data$Outcome <- factor(data$Outcome)

str(data)

data %>%
  group_by(Outcome)%>%
  summarise(per = n()/nrow(data))%>%
  ggplot(aes(x=Outcome, y=per, fill = Outcome)) + 
  geom_bar(stat='identity') +
  geom_text(aes(label = round(per, 2)), vjust = 2)
# the data is almost balanced with 65 percent of 0 and 35 percent of 1
# data shows low correlation between independent variables

#training and testing
#set initial seed
set.seed(8)

# create a boolean flag to split data
# sample.split from caTools
splitData = sample.split(data$Outcome, SplitRatio = 0.7)

#split_data
# create train and test datasets
train_set = data[splitData,]
train_set
nrow(train_set)/nrow(data)
dim(train_set)


test_set = data[!splitData,]
test_set
dim(test_set)
nrow(test_set)/nrow(data)


# use train to create our model
# use all independent variables 
model = glm(Outcome ~ ., data = train_set, family = binomial)
summary(model)
#AIC is 523.2, with SkinThickness and Age being non significant in the model
# predict on the train set
trainPredict = predict(model, newdata = train_set, 
                       type = 'response')

# assign 0s or 1s for the values
p_class = ifelse(trainPredict > 0.5, 1,0)
confusionMatrix(table(p_class,train_set$Outcome), positive='1') # Accuracy = 0.7807


# predict on the test set
testPredict = predict(model, newdata = test_set, type = 'response')

# assign 0s or 1s for the values
p_class = ifelse(testPredict > 0.5, 1,0)

confusionMatrix(table(p_class,test_set$Outcome), positive='1') # Accuracy = 0.7652

#############################

# lift chart
#prediction from ROCR library
pred = prediction(trainPredict, train_set$Outcome )
perf = performance( pred, "lift", "rpp" ) #RPP=Rate of positive prediction 
plot(perf, main="lift curve", xlab = 'Proportion of Patients (sorted prob)')

#roc.curve function from ROSE package
roc.curve(train_set$Outcome,trainPredict)  # AUC = 0.840

# check for multi-collinearity
vif(model)
# all vifs are lower than 5

#model selection - try to improve on model

model.imp = glm(Outcome ~ . -Age -SkinThickness, data = train_set, family = binomial)
summary(model.imp)  # AIC = 519.46 slightly improve

# predict on the train set
trainPredict.imp = predict(model.imp, newdata = train_set, 
                       type = 'response')

# assign 0s or 1s for the values
p_class = ifelse(trainPredict.imp > 0.5, 1,0)
confusionMatrix(table(p_class,train_set$Outcome), positive='1') # Accuracy = 0.7788 not so much difference


# predict on the test set
testPredict.imp = predict(model.imp, newdata = test_set, type = 'response')

# assign 0s or 1s for the values
p_class = ifelse(testPredict.imp > 0.5, 1,0)

confusionMatrix(table(p_class,test_set$Outcome), positive='1') # Accuracy = 0.7609

# lift chart
#prediction from ROCR library
pred = prediction(trainPredict.imp, train_set$Outcome )
perf = performance( pred, "lift", "rpp" ) #RPP=Rate of positive prediction 
plot(perf, main="lift curve", xlab = 'Proportion of Patients (sorted prob)')

#roc.curve function from ROSE package
roc.curve(train_set$Outcome,trainPredict.imp) # AUC = 0.840

#Based on the logistic model, Pregnancies, Glucose, BloodPressure, Insulin, DiabetesPedigreeFunction and Age were all significant contributors in the model.
#The model training accuracy and testing accuracy is 0.7807 and 0.7609 respectively
#This suggests that the model is not over/underfitted
#Additionally, the lift curve shows a high value in the early segments, and AUC for the ROC curve is 0.840
#This shows that the model is good at distinguishing between patients with and without disease