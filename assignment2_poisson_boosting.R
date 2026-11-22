# Assignment 2, Question 2 (original R submission, Dec 2024)
# Poisson boosting machine on freMTPL2freq: M = 50 trees of depth 3,
# at least 10,000 policies per leaf, no shrinkage.
# Adapted from the course code accompanying Wuthrich & Merz (2023).
# Only file paths were changed for this repository.

load("freMTPL2freq.rda")
######### choice of learning and test sample
set.seed(100)
ll <- sample(c(1:nrow(freMTPL2freq)), round(0.9*nrow(freMTPL2freq)), replace = FALSE) 
learn <- freMTPL2freq[ll,]
test <- freMTPL2freq[-ll,]
n_l = nrow(learn)
n_t = nrow(test)

#########  Poisson boosting machine
library(rpart)
J0 <- 3      #depth of tree
M0 <- 50      #iterations
learn$fit0 <- learn$Exposure
test$fit0  <- test$Exposure
for (m in 1:M0){
  PBM.1 <- rpart(cbind(fit0,ClaimNb) ~ Area + VehPower + VehAge + DrivAge 
                 + BonusMalus + VehBrand + VehGas + Density + Region, 
                 data=learn, method="poisson",
                 control=rpart.control(maxdepth=J0, maxsurrogate=0, xval=1, 
                                       minbucket=10000, cp=0.00005))     
  learn$fit0 <- learn$fit0 * predict(PBM.1)
  learn[,paste("PBM_",m, sep="")] <-  learn$fit0
  test$fit0 <- test$fit0 * predict(PBM.1, newdata=test)
  test[,paste("PBM_",m, sep="")] <-  test$fit0
}

# in-sample loss and out-of-sample loss
losses <- array(NA, c(2,M0))
for (m in 1:M0){
  losses[1,m] <- 200*(sum(learn[,paste("PBM_",m, sep="")])-sum(learn$ClaimNb)+
                        sum(log((learn$ClaimNb/learn[,paste("PBM_",m, sep="")])^
                                  (learn$ClaimNb))))/n_l
  losses[2,m] <- 200*(sum(test[,paste("PBM_",m, sep="")])-sum(test$ClaimNb)+
                        sum(log((test$ClaimNb/test[,paste("PBM_",m, sep="")])^
                                  (test$ClaimNb))))/n_t
}

losses[,M0] 

plot(x=c(0:M0), y=c(32.93518, losses[1,]), type='l', col="red", ylim=c(30,33.5),
     xlab="number of iterations", ylab="average in-sample loss (in 10^(-2))", 
     main=paste("decrease of in-sample loss (depth=", J0,")", sep=""))
points(x=c(0:M0), y=c(32.93518, losses[1,]), pch=19, col="red")
plot(x=c(1:M0), y=losses[2,], type='l', lwd=2, col="red", ylim=c(30,31.5), 
     xlab="number of iterations", ylab="average out-of-sample loss (in 10^(-2))", 
     main="decrease of out-of-sample loss")
