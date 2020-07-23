#https://www.r-bloggers.com/how-to-perform-a-logistic-regression-in-r/
#https://stats.stackexchange.com/questions/306267/is-mse-decreasing-with-increasing-number-of-explanatory-variables
#https://cran.r-project.org/web/packages/glmnet/vignettes/glmnet.pdf
setwd("E:/Workshop/Machine Learning")

### Assignment 1
spambase = read.csv("Data/spambase.csv", header = TRUE)

n=dim(spambase)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=spambase[id,]
test=spambase[-id,]

logistic <- function(data, p) {

  model <- glm(as.factor(Spam) ~.,family=binomial, data=train)

  # type "response" is used to give the probabilities instead of log-odds
  predicted  <- predict(model, data, type = 'response') 
  classified <- ifelse(predicted > p,1,0)

  con_matrix <- table(Predicted = classified, Actual = data$Spam)

  miss_rate  <- 1- sum(diag(con_matrix))/sum(con_matrix)
  #miss_rate <- mean(classified != data$Spam)

  result = list("Confusion Matrix" = con_matrix, "Missclassification Rate" = miss_rate)
  result
}

logistic(train, 0.5)
logistic(test, 0.5)
logistic(train, 0.8)
logistic(test, 0.8)


library(kknn)
knn <- function(data, k){
  
  classified <- kknn(formula = as.factor(Spam) ~ ., 
                     train = train, 
                     test = data,
                     k = k)
  
  con_matrix <- table(Predicted = classified$fitted.values, Actual = data$Spam)
  
  miss_rate <- 1- sum(diag(con_matrix))/sum(con_matrix)
  
  result = list("Confusion Matrix" = con_matrix,
                "Missclassification Rate" = miss_rate)
  return(result)
}

knn(train, 30)
knn(test, 30)
knn(train, 1)
knn(test, 1)


### Assignment 2
library(readxl)

machines <- read_excel("Data/machines.xlsx")

log_like <- function(x, theta) {
  sum(log(theta * exp(-theta * x)))
}

thetas <- seq(0, 5, by = 0.1)
logs   <- sapply(thetas, function(x) {sum(log_like(x = machines$Length, theta = x))})

max_theta <- thetas[which.max(logs)]


length <- hist(machines$Length, plot=FALSE)
m   <- length$counts / length$density
m   <- max(m[which(!is.nan(m))])
d   <- density(machines$Length)
d$y <- d$y * m

plot(length, 
     col  = "#00CCFF", 
     main = "Machine Distribution",
     xlab = "Lifetime", 
     ylab = "Frequency", 
     xlim = c(0, 5))
lines(d, 
      col = "#FF3366", 
      lwd = 2)



plot(thetas, 
     logs, 
     main = "Log-Likelihood", 
     col  = "red",
     xlab = "Theta", 
     ylab = "Log-Likelihood", 
     type = "p",
     lwd  = ifelse(thetas == thetas[which.max(logs)],3, 1),
     cex  = ifelse(thetas == thetas[which.max(logs)],1.5, 1))

set.seed(12345)
sample <- sample(x = machines$Length, size = 6)
sample_logs <- sapply(thetas, function(x) {sum(log_like(l = sample, theta = x))})

plot(thetas, 
     logs, 
     col = "red",
     main = "Log-Likelihood", 
     xlab = "Theta", 
     ylab = "Log-Likelihood",
     type = "p", 
     lwd  = 1, 
     ylim = c(-115,0))
points(thetas, 
       sample_logs, 
       col = "blue", 
       lwd = 1)


posterior <- function(theta, x){
  lambda <- 10
  prior <- log(lambda * exp(-lambda * thetas))

  posteriors <- prior + logs
  return((posteriors))
}
posteriors <- posterior(thetas, machines$Length)

# prior <- function(theta) {
#   lambda <- 10
#   lambda * exp(-lambda * theta)
# }
# 
# posteriors <- sapply(1:length(thetas), function(x) {logs[x] + log(prior(thetas[x]))})


plot(thetas, 
     posteriors, 
     col  = "green",
     main = "Log-Posterior", 
     xlab = "Theta", 
     ylab = "Log-Posterior",
     type = "p", 
     lwd  = 1)
points(thetas, 
       logs, 
       col = "red", 
       lwd = 1)


set.seed(12345)
new_obs <- rexp(50, max_theta)

par(mfrow=c(1, 2))
hist(new_obs, 
     breaks = 10, 
     main   = "Generated Data",
     xlab   = "Lifetime", 
     ylab   = "Frequency", 
     xlim   = c(0, 5), 
     col    = "green",
     ylim   = c(0,25))

hist(machines$Length, 
     breaks = 10, 
     main   = "Original Data",
     xlab   = "Lifetime", 
     ylab   = "Frequency", 
     col    = "red",
     xlim   = c(0,5),
     ylim   = c(0,25))



### Assignment 3

data("swiss")
library(ggplot2)

mylin=function(X,Y, Xpred){
  Xpred1=cbind(1,Xpred)
  X= cbind(1,X)
  beta <- (solve(t(X)%*%X))%*%(t(X)%*%Y)
  Res=Xpred1%*%beta
  return(Res)
}

myCV=function(X,Y,Nfolds){
  n=length(Y)
  p=ncol(X)
  set.seed(12345)
  ind=sample(n,n)       # Shuffle
  X1=X[ind,]            # New X after shuffle
  Y1=Y[ind]             # New Y after shuffle
  sF=floor(n/Nfolds)    # Fold size
  MSE=numeric(2^p-1)
  Nfeat=numeric(2^p-1)
  Features=list()
  curr=0
  
  #we assume 5 features.
  
  for (f1 in 0:1)
    for (f2 in 0:1)
      for(f3 in 0:1)
        for(f4 in 0:1)
          for(f5 in 0:1){
            model= c(f1,f2,f3,f4,f5)                # Selected Features
            if (sum(model)==0) next()               # Skips the first iteration where the model have no feature
            SSE=0
            selected_feature <- which(model == 1)
            for (k in 1:Nfolds){
              breaks <- seq(1,n,sF)
              folds <- ind[breaks[k]:breaks[k+1]-1]
              
              Xtrain <- X1[-folds,selected_feature]
              Xpred  <- X1[folds,selected_feature]
              Ytrain <- Y1[-folds]
              Ypred  <- Y1[folds]
              
              Yp <- mylin(Xtrain,Ytrain,Xpred)
              SSE=SSE+sum((Ypred-Yp)^2)
            }
            curr=curr+1
            MSE[curr]=SSE/n
            Nfeat[curr]=sum(model)
            Features[[curr]]=model
            
          }

  plot(Nfeat, MSE, xlab = "Number of features", ylab = "MSE")
  
  i=which.min(MSE)
  return(list(CV=MSE[i], Features=Features[[i]]))
}

myCV(as.matrix(swiss[,2:6]), swiss[[1]], 5)

library(psych)
pairs.panels(swiss)



##### Assignment 4
library(ggplot2)
library(glmnet)
library(MASS)

tecator <- read.csv("Data/tecator.csv", header = TRUE)

ggplot(tecator, aes(x = Protein, y = Moisture)) +
  geom_point(alpha = 0.5, color = 'red') +
  geom_smooth(method = "lm", color = "blue")

n=dim(tecator)[1]
set.seed(12345)
id=sample(1:n, floor(n*0.5))
train=tecator[id,]
test=tecator[-id,]

M <- function(data,power) {
  MSE <- matrix(0,power,1)
  
  for (i in 1:power) {
    model <- lm(Moisture ~ poly(Protein,i), data = train)
    pred  <- predict(model, data)
    MSE[i,] <- mean((data$Moisture - pred)^2)
    }

  return(MSE)
}
train_MSE <- M(train,6)
test_MSE  <- M(test,6)


df = data.frame(train_MSE,test_MSE)

round(df,2)

ggplot(df) + 
  geom_line(aes(x = 1:6, y = train_MSE, color = "train_MSE"), size = 1) +
  geom_line(aes(x = 1:6, y = test_MSE, color = "test_MSE"), size = 1) + 
  ylab("MSE") + 
  xlab("Model complexity")

#Using stepAIC on all data
channels_all_data   <- tecator[,2:101] 
Fat_model_all_data  <- lm(tecator$Fat ~ ., channels_all_data)
best_fit_all_data   <- stepAIC(Fat_model_all_data, trace=FALSE)

as.matrix(c("Selected Channels"= length(best_fit_all_data$coefficients)))

#Using stepAIC on training data only
channels_train_data   <- train[,2:101] 
Fat_model_train_data  <- lm(train$Fat ~ ., channels_train_data)
best_fit_train_data   <- stepAIC(Fat_model_train_data, trace=FALSE)

as.matrix(c("Selected Channels"= length(best_fit_train_data$coefficients)))

#Testing stepAIC model on the test data
AIC_model  <- lm(formula(best_fit_train_data), channels_train_data)
pred  <- predict(AIC_model, test)
MSE   <- as.matrix(c("MSE" = mean((test$Fat - pred)^2)))
round(MSE,2)


ridge <- glmnet(x = as.matrix(tecator[,2:101]), 
                y = as.matrix(tecator$Fat), 
                alpha = 0, 
                family = "gaussian")

plot(ridge, xvar="lambda", label=TRUE)

lasso <- glmnet(x = as.matrix(tecator[,2:101]), 
                y = as.matrix(tecator$Fat), 
                alpha = 1, 
                family = "gaussian")

plot(lasso, xvar="lambda", label=TRUE)

set.seed(12345)
lambda_selection <- 10^seq(-5,5,0.1)
lasso_cv         <- cv.glmnet(x = as.matrix(tecator[,2:101]), 
                        y = as.matrix(tecator$Fat), 
                        alpha=1, 
                        family="gaussian",
                        lambda = lambda_selection) 

plot(lasso_cv)
as.matrix(c("Optimum Lambda" = lasso_cv$lambda.min))
round(as.matrix(c("MSE" = lasso_cv$cvm[which.min(lasso_cv$cvm)])),2)

