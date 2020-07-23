## References 
#https://stats.stackexchange.com/questions/82323/shrinkage-parameter-in-adaboost 
#https://xavierbourretsicotte.github.io/AdaBoost.html 
#https://stats.stackexchange.com/questions/163747/rpart-classification-why-is-my-predict-output-not-adhering-to-type-class 
#https://www.youtube.com/watch?v=6EXPYzbfLCE 
#https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/ 

setwd('D:/Machine Learning/Workshop/Machine Learning/Block 2/Lab 1 Block 2')
RNGversion('3.5.1')

library(mboost)
library(randomForest)
library(ggplot2)

### 1. ENSEMBLE METHODS
sp      <- as.data.frame(read.csv2("Data/spambase.csv"))
sp$Spam <- as.factor(sp$Spam)

n=dim(sp)[1]
set.seed(12345)
id=sample(1:n, floor(n*2/3))
train=sp[id,]
test=sp[-id,]

k         <- seq(10,100,10)
miss_rate <- matrix(0,10,11)
colnames(miss_rate) <- c("L=0.1", "L=0.2", "L=0.3", "L=0.4", "L=0.5", "L=0.6", "L=0.7", "L=0.8", "L=0.9", "L=1", "R.F.")
rownames(miss_rate) <- c("10 Ts", "20 Ts", "30 Ts", "40 Ts", "50 Ts", "60 Ts", "70 Ts", "80 Ts", "90 Ts", "100 Ts")

# Adabtive Boosting
for(i in k){
  for(j in k){
    ada_model         <- blackboost(train$Spam ~ ., 
                                    data    = train, 
                                    family  = AdaExp(), 
                                    control = boost_control(mstop = i, nu = j/100))
    
    ada_pred          <- predict(ada_model, 
                                 newdata = test, 
                                 type    ="class")
    
    ada_mat           <- table(Predected = ada_pred, Actual = test$Spam)
    
    miss_rate[i/10,j/10] <- round((1 - sum(diag(ada_mat))/sum(ada_mat)),4)
  }
}

Ada_Best_Model    <- blackboost(train$Spam ~ ., 
                            data    = train, 
                            family  = AdaExp(), 
                            control = boost_control(mstop = 100))

ada_best_pred     <- predict(Ada_Best_Model, 
                             newdata = test, 
                             type    ="class")

ada_best_mat      <- table(Predected = ada_best_pred, Actual = test$Spam)
round((1 - sum(diag(ada_best_mat))/sum(ada_best_mat)),4)

# Random Forest
for(i in k){
  rf_model           <- randomForest(train$Spam ~ ., 
                                     data  = train, 
                                     ntree = i,
                                     importance = TRUE,
                                     proximity = TRUE)

  rf_pred            <- predict(rf_model, 
                               newdata = test, 
                               type    ="class")

  rf_mat             <- table(Predected = rf_pred, Actual    = test$Spam)
  miss_rate[i/10,11] <- round((1 - sum(diag(rf_mat))/sum(rf_mat)),4)
}

list("Adaboost Least Error" = miss_rate[which.min(miss_rate[,1:10])], 
     "Adaboost Confusion Matrix" = ada_best_mat, 
     "Random Forest Least Error" = miss_rate[which.min(miss_rate[,11])+100],
     "Random Forest Confusion Matrix" = rf_mat, 
     "Error Rates" = miss_rate)



par(mfrow = c(1, 2)) 
plot(as.vector(t(miss_rate[,1:10])), 
     main = "Adaboost Error Rate", 
     xlab = "Number of Trees", 
     ylab = "Error Rate", 
     ylim = c(0.04,0.13),
     type = "l", 
     pch  = 19, 
     col  = "red")
plot(miss_rate[,11],
     main = "Random Forest Error Rate", 
     xlab = "Number of Trees (in 10s)", 
     ylab = "Error Rate",
     ylim = c(0.04,0.13),
     type = "l", 
     pch  = 19,
     col  = "blue")






# No. of variables tried at each split: 7 

plot(rf_model)

# How pure the nodes are at the end of the tree without each variable
varImpPlot(rf_model, main = "Variable Importance", pch = 19, n.var = 7)

# How many nodes are too often used
hist(treesize(rf_model))

# How many times each variable been used
which(varUsed(rf_model))

sort(varUsed(rf_model))



# View single tree
getTree(rf_model,1)

MDSplot(rf_model, train[,58])

tuneRF(test[,-58], test[,58],
       stepFactor = 0.25,
       plot = TRUE,
       ntreeTry = 100,
       trace = TRUE,
       improve = 0.01)


dist_matrix <- dist(1-rf_model$proximity)
MDS <- cmdscale(dist_matrix, eig = TRUE, x.ret = TRUE)

var <- round(MDS$eig/sum(MDS$eig)*100, 1)

mds_values <- MDS$points

mds_data <- data.frame(smaple=rownames(mds_values), X = mds_values[,1], Y = mds_values[,2], Status = data.imputed$hd)

### 2. MIXTURE MODELS



set.seed(1234567890)
max_it <- 100 # max number of EM iterations
min_change <- 0.1 # min change in log likelihood between two consecutive EM iterations
N=1000 # number of training points
D=10 # number of dimensions
x <- matrix(nrow=N, ncol=D) # training data
true_pi <- vector(length = 3) # true mixing coefficients
true_mu <- matrix(nrow=3, ncol=D) # true conditional distributions
true_pi=c(1/3, 1/3, 1/3)
true_mu[1,]=c(0.5,0.6,0.4,0.7,0.3,0.8,0.2,0.9,0.1,1)
true_mu[2,]=c(0.5,0.4,0.6,0.3,0.7,0.2,0.8,0.1,0.9,0)
true_mu[3,]=c(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)
plot(true_mu[1,], 
     type="o", 
     col="blue", 
     ylim=c(0,1), 
     main = "Original Data", 
     xlab = "Number of Dimensions", 
     ylab = "True Mu")
points(true_mu[2,], type="o", col="red")
points(true_mu[3,], type="o", col="green")
# Producing the training data
for(n in 1:N) {
  k <- sample(1:3,1,prob=true_pi)
  for(d in 1:D) {
    x[n,d] <- rbinom(1,1,true_mu[k,d])
  }
}

EM <- function(c){
K=c # number of guessed components
z <- matrix(nrow=N, ncol=K) # fractional component assignments
pi <- vector(length = K) # mixing coefficients
mu <- matrix(nrow=K, ncol=D) # conditional distributions
llik <- vector(length = max_it) # log likelihood of the EM iterations
# Random initialization of the paramters
pi <- runif(K,0.49,0.51)
pi <- pi / sum(pi)
for(k in 1:K) {
  mu[k,] <- runif(D,0.49,0.51)
}
pi
mu

for(it in 1:max_it) {

  #plot(mu[1,], type="o", col="blue", ylim=c(0,1))
  #points(mu[2,], type="o", col="red")
  #points(mu[3,], type="o", col="green")
  #points(mu[4,], type="o", col="yellow")
  
  #Sys.sleep(0.5)
  # E-step: Computation of the fractional component assignments (responsiblities)
  # Your code here
  for (i in 1:N) {
    phi = c()
    for (k in 1:K) {
      y1 = mu[k,]^x[i,]
      y2 = (1- mu[k,])^(1-x[i,])
      phi = c(phi, prod(y1,y2))
    }
    z[i,] = (pi*phi) / sum(pi*phi) 
  } 
 
  #Log likelihood computation.
  # Your code here

  likelihood <-matrix(0,1000,K)
  llik[it] <-0
  for(n in 1:N)
  {
    for (k in 1:K)
    {
      likelihood[n,k] <- pi[k]*prod( ((mu[k,]^x[n,])*((1-mu[k,])^(1-x[n,]))))
    }
    llik[it]<- sum(log(rowSums(likelihood)))
  }

  cat("iteration: ", it, "log likelihood: ", llik[it], "\n")
  flush.console()
  # Stop if the lok likelihood has not changed significantly
  # Your code here
  if (it > 1)
  {
    if (llik[it]-llik[it-1] < min_change)
    {
      break()
    }
  } 
  
  #M-step: ML parameter estimation from the data and fractional component assignments
  # Your code here
  mu<- (t(z) %*% x) /colSums(z)
  # N - Total no. of observations
  pi <- colSums(z)/N
  
  if(K == 2)
  {
    plot(mu[1,], 
         type="o", 
         col="blue", 
         ylim=c(0,1),
         main = "K = 2", 
         xlab = "Number of Dimensions", 
         ylab = "True Mu")
    points(mu[2,], type="o", col="red")
  }
  else if(K==3)
  {
    plot(mu[1,],            
         type="o", 
         col="blue", 
         ylim=c(0,1),
         main = "K = 3", 
         xlab = "Number of Dimensions", 
         ylab = "True Mu")
    points(mu[2,], type="o", col="red")
    points(mu[3,], type="o", col="green")
  }
  
  else
  {
    plot(mu[1,],            
         type="o", 
         col="blue", 
         ylim=c(0,1),
         main = "K = 4", 
         xlab = "Number of Dimensions", 
         ylab = "True Mu")
    points(mu[2,], type="o", col="red")
    points(mu[3,], type="o", col="green")
    points(mu[4,], type="o", col="yellow")
  }
}
pi
mu
plot(llik[1:it], type="o")
}
EM(2)
EM(3)
EM(4)





