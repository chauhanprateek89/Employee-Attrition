# IBM Employee Attrition analysis
# by Shuma Ishigami

# Prediction of Attrition by using SVM

# install.packages("e1071")
library("e1071")

# Importing data
d<-read.csv("HR_Attrition.csv")

# Data cleaning
# to see a structure of data
# str(d)
# remove EmployeeCount,Over18 and StandardHours
# because they contin only one value or factor
d$EmployeeCount<-NULL
d$Over18<-NULL
d$StandardHours<-NULL


# Classification by SVM
# Creat data for training and test
set.seed(2017)
tr.number<-sample(nrow(d),nrow(d)/2) 
d.train<-d[tr.number,]
d.test<-d[-tr.number,]

# base model
svm.model.1<-svm(
  d.train$Attrition~.,
  type="C-classification",
  data=d.train
)

# Prediction table for training data
t=table(d.train$Attrition,
        predict(svm.model.1,d.train[,-2]))/nrow(d.train)
sum(diag(t)) # Correct rate 
# Prediction for test data
t=table(d.test$Attrition,
        predict(svm.model.1,d.test[,-2]))/nrow(d.test)
sum(diag(t))


# 3-fold cross validation 
svm.model<-svm(
  Attrition~.,
  type="C-classification",
  cost=3,
  data=d,
  cross=3
)
summary(svm.model)

# tuning parameters
# First search 
gamma.range<-10^(-3:3)
cost.range<-10^(-2:2)
tuning<-tune.svm(
  Attrition~.,
  type="C-classification",
  gamma=gamma.range,
  cost=cost.range,
  data=d,
  tunecontrol=tune.control(sampling="cross",cross=3)
)
tuning$best.parameters # best parameters at this point 
# gamma=0.001=10^(-3) and cost=100=10^2

1-tuning$best.performance # accuracy 

# plot of performance grid
plot(tuning, transform.x=log10, transform.y=log10)
# the darker the colour is, there is a 
#higher probability of 
# existance of optimum parameters
# We will explore around a diamond shape place in the plot 
# gamma around 10^(-2) and cost about 10^(1)

# we do not search the left-upper couner because
# as cost parameter hikes, it results in a better 
# one-shot prediction but usually this is over-fitting


# Second search
gamma.range<-10^seq(-3,-1,length=10) # -2 +-1
cost.range<-10^seq(0,2,length=10) # 1 +-1
tuning<-tune.svm(
  Attrition~.,
  type="C-classification",
  gamma=gamma.range,
  cost=cost.range,
  data=d,
  tunecontrol=tune.control(sampling="cross",cross=3)
)
tuning$best.parameters 
plot(tuning, transform.x=log10, transform.y=log10)
# now we get best parameters 
# gamma=0.001668101 and cost=35.93814
# using this result 
tuned.model<-svm(
  Attrition~.,
  type="C-classification",
  gamma=0.001668101,
  cost=35.93814,
  data=d,
  cross=3
)
summary(tuned.model)
# accuracy improved