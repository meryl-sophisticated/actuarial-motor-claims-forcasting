# Assignment 2, Question 1 (original R submission, Dec 2024)
# Poisson regression tree on freMTPL2freq: large tree (cp = 0.00005,
# at least 1,500 policies per leaf), pruned by 10-fold cross-validation.
# Transcribed from the submitted assignment; file path and plot labels
# changed for this repository. Adapted from the course code accompanying
# Wuthrich & Merz (2023).

library(rpart)
library(rpart.plot)
library(plotrix)
library(Hmisc)

load("freMTPL2freq.rda")

set.seed(100)
ll <- sample(1:nrow(freMTPL2freq), round(0.9 * nrow(freMTPL2freq)), replace = FALSE)
learn <- freMTPL2freq[ll, ]
test <- freMTPL2freq[-ll, ]
n_l <- nrow(learn)
n_t <- nrow(test)

# (a) very large SBS regression tree
tree_large <- rpart(
  cbind(Exposure, ClaimNb) ~ Area + VehPower + VehAge + DrivAge +
    BonusMalus + VehBrand + VehGas + Density + Region,
  data = learn, method = "poisson",
  control = rpart.control(cp = 0.00005, minbucket = 1500, xval = 1)
)
rpart.plot(tree_large)

# (b) 10-fold cross-validation for cost-complexity pruning
K <- 10
tree_cv <- rpart(
  cbind(Exposure, ClaimNb) ~ Area + VehPower + VehAge + DrivAge +
    BonusMalus + VehBrand + VehGas + Density + Region,
  data = learn, method = "poisson",
  control = rpart.control(cp = 0.00005, minbucket = 1500, xval = K)
)
printcp(tree_cv)

set.seed(100)
xgroup <- rep(1:K, length = nrow(learn))
xfit <- xpred.rpart(tree_cv, xgroup)
n_subtrees <- dim(tree_cv$cptable)[1]
std3 <- numeric(n_subtrees)
err3 <- numeric(n_subtrees)
err_group <- numeric(K)
for (i in 1:n_subtrees) {
  for (k in 1:K) {
    ind_group <- which(xgroup == k)
    err_group[k] <- 2 * sum(learn[ind_group, "Exposure"] * xfit[ind_group, i] - learn[ind_group, "ClaimNb"] +
      log((learn[ind_group, "ClaimNb"] / (learn[ind_group, "Exposure"] * xfit[ind_group, i]))^learn[ind_group, "ClaimNb"]))
  }
  err3[i] <- mean(err_group)
  std3[i] <- sd(err_group)
}

y1 <- err3 * K / n_l
s1 <- K * std3 / n_l
x1 <- log10(tree_cv$cptable[, 1])
errbar(x = x1, y = y1 * 100, yplus = (y1 + s1) * 100, yminus = (y1 - s1) * 100,
       xlim = rev(range(x1)), col = "blue", main = "cross-validation error plot",
       ylab = "CV error (in 10^(-2))", xlab = "cost-complexity parameter (log-scale)")
lines(x = x1, y = y1 * 100, col = "blue")
abline(h = min(y1 + s1) * 100, lty = 1, col = "orange")
abline(h = min(y1) * 100, lty = 1, col = "magenta")
legend(x = "topright", col = c("blue", "orange", "magenta"), lty = c(1, 1, 1), lwd = c(1, 1, 1),
       pch = c(19, -1, -1), legend = c("tree", "1-SD rule", "min. CV rule"))

optimal_cp <- tree_cv$cptable[which.min(tree_cv$cptable[, "xerror"]), "CP"]
tree_pruned <- prune(tree_cv, cp = optimal_cp)
rpart.plot(tree_pruned)

# (c) in-sample and out-of-sample loss of the best pruned tree
Poisson.Deviance <- function(pred, obs) {
  2 * (mean(pred) - mean(obs) + mean(log((obs / pred)^obs)))
}
learn$fit <- predict(tree_pruned) * learn$Exposure
test$fit <- predict(tree_pruned, newdata = test) * test$Exposure
cat("in-sample loss:", Poisson.Deviance(learn$fit, learn$ClaimNb), "\n")
cat("out-of-sample loss:", Poisson.Deviance(test$fit, test$ClaimNb), "\n")
