############################################## Linear Model ##########################
rm(list = ls()) 
graphics.off()


library(ISLR)

MyData <- read.csv(file="C:/Users/Matan/Desktop/MSA-MRM/Classes/MSA 8150 - Machine Learning for Analytics/FinalProject/data_music/Train.csv", header=TRUE,  sep=",")

splitting = 'random'

if (splitting == 'random'){
  rand.ind = sample(seq(1:959),959)
  MyData.train = MyData[rand.ind[1:767],]
  MyData.test = MyData[rand.ind[768:959],]
}else{
  MyData.train = MyData[1:767,]
  MyData.test = MyData[768:959,]
}


y1 = MyData.train[,'latitude']
y2 = MyData.train[,'longitude']
y1
#y3 = MyData.test[,'latitude']
#y4 = MyData.test[,'longitude']

cor(MyData.train, use="complete.obs", method="kendall") 
#cov(MyData.train, use="complete.obs")
## Using Forward Selection:

linReg <- lm(cbind(y1, y2) ~ MyData.train$Var5 + MyData.train$Var20 + MyData.train$Var47 + MyData.train$Var51 + MyData.train$Var52 + MyData.train$Var65 + MyData.train$Var69 + MyData.train$Var82 + MyData.train$Var85 + MyData.train$Var93 + MyData.train$Var107, data = MyData.train)
lm_summary <- summary(linReg)
print(lm_summary) ## Best R_squared =  0.2912
mean(linReg$residuals^2)

predictions <- predict(linReg, MyData.test, interval = 'prediction')
as.data.frame(predictions)

y_hat <- predictions[,1]
y_hat

y <- cbind(MyData.test$latitude,MyData.test$longitude)
y <- as.data.frame(y)
y

euc_distance <- sqrt(sum((y_hat - y)^2))
euc_distance # E = 779.0749

mse <- mean((linReg$residuals)^2)
mse #MSE for linear regression = 1087.269


############################################## Ridge, Lasso Models ##########################



library(ISLR)
library(glmnet)
library(pls)

y3 = MyData[,'latitude']
y4 = MyData[,'longitude']

x=model.matrix(cbind(y3,y4)~.,MyData)[,-1]
y = as.data.frame(cbind(y3,y4))
#y = as.numeric(unlist(y))

# a grid for the possible values of lambda
grid=10^seq(10,-2,length=100)

# alpha = 0 --> Ridge, alpha = 1 --> LASSO
ridge.mod=glmnet(x,y,alpha=0,lambda=grid, family = "mgaussian")

# comparing the magnitude of the coefficients for two different lambdas
print(ridge.mod$lambda[50]) #11497.57
print(coef(ridge.mod)[,50])
print(sqrt(sum(coef(ridge.mod)[-1,50]^2)))


print(ridge.mod$lambda[60]) #705.4802
print(coef(ridge.mod)[,60])
print(sqrt(sum(coef(ridge.mod)[-1,60]^2)))


# Model selection:
set.seed(1)
train=sample(1:nrow(x), nrow(x)/2)
test=(-train)
y.test=y[test]

ridge.mod=glmnet(x[train,],y[train],alpha=0,lambda=grid, thresh=1e-12)

set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)

bestlam=cv.out$lambda.min
print(bestlam)
ridge.pred=predict(ridge.mod,s=bestlam,newx=x[test,])
print(mean((ridge.pred-y.test)^2)) # mse = 2659.55

# what if we did not do the Ridge? (lambda=0)
ridge.pred=predict(ridge.mod,s=0,newx=x[test,])
print(mean((ridge.pred-y.test)^2)) #mse = 4027.061

##RIDGE IS WAY BETTER

##################################LASSO REGRESSION########################
##################################LASSO REGRESSION########################

lasso.mod=glmnet(x[train,],y[train],alpha=1,lambda=grid)
plot(lasso.mod)
set.seed(1)
cv.out=cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam=cv.out$lambda.min
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
print(mean((lasso.pred-y.test)^2))
# what would have happened if no regularization?
lasso.pred.0=predict(lasso.mod,s=0,newx=x[test,])
print(mean((lasso.pred.0-y.test)^2))

# we want to see the sparsity of the solutions 
lasso.mod.full=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(lasso.mod.full,type="coefficients",s=bestlam)[1:20,]
print(lasso.coef)

##################################PCR########################
##################################PCR########################

set.seed(2)
# option scale does the standardizing
# when we set validation="CV" then pcr performs 10-fold CV for each number of possible components
pcr.fit=pcr(Salary~., data=Hitters,scale=TRUE,validation="CV")
print(summary(pcr.fit))
# plotting the CV curves
validationplot(pcr.fit,val.type="MSEP")
set.seed(1)
# focusing only on the training data above
pcr.fit=pcr(Salary~., data=Hitters,subset=train,scale=TRUE, validation="CV")
validationplot(pcr.fit,val.type="MSEP")
# since the lowest CV happens at M=7:
pcr.pred=predict(pcr.fit,x[test,],ncomp=7)
print(mean((pcr.pred-y.test)^2))# pcrMSE:96556.22
