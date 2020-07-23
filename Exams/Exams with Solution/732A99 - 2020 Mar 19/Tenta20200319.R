######### Only for 732A99

# Question: Show that the bagging error equals 1/B of the average individual error, when the
# individual errors have zero mean and are uncorrelated. That is, for B=3:
# E[((e_1+e_2+e_3)/3)^2] = ((E[e_1^2] + E[e_2^2] + E[e_3^2])/3)/3
# where e_b is short for \epsilon^b(x).

# Answer: It suffices to prove the following:

# E[(e_1+e_2+e_3)^2] = E[e_1 e_1] + E[e_1 e_2] + E[e_1 e_3] + E[e_2 e_2] + E[e_2 e_3] + E[e_3 e_3]
# = E[e_1 e_1] + E[e_2 e_2] + E[e_3 e_3]
# + E[(e_1 - E[e_1]) (e_2 - E[e_2]] + E[(e_1 - E[e_1]) (e_3 - E[e_3])] + E[(e_2 - E[e_2]) (e_3 - E[e_3])]
# = E[e_1 e_1] + E[e_2 e_2] + E[e_3 e_3]

# Note that E[e_1 e_2] = E[(e_1 - E[e_1]) (e_2 - E[e_2]]) because the individual errors have
# zero mean, and E[(e_1 - E[e_1]) (e_2 - E[e_2]]) = 0 because the individual errors are
# uncorrelated. For more details, see slide 7 in the lecture on ensemble methods.

#########

# Question: Show the above experimentally.

# Answer: Note that the error involves an expectation over X. Therefore, it is difficult to
# calculate exactly. We approximate it by sampling and averaging the errors for 100 X points.

library(mvtnorm)
n <- 10

be <- NULL
aie <- NULL

for(i in 1:100){
  sigma <- matrix(data = 0, ncol=n) # off-diagonals set to zero because errors are uncorrelated
  sigma <- diag(runif(n,1,2))
  x <- rmvnorm(n=1, mean=rep(0,n), sigma=sigma)

  be <- c(be,mean(x)^2) # bagging error
  aie <- c(aie,mean(x^2)) # average individual error
}

hbe <- hist(be)
haie <- hist(aie)
plot(hbe, col=rgb(0,0,1,1/4), xlim=c(0,5)) # bagging error
plot(haie, col=rgb(1,0,0,1/4), xlim=c(0,5), add=T) # average individual error

#########

# Question: Do the same with the variance. Also, answer why a low variance is desirable.

# Answer: Low variance means stability, i.e. insensitivity to the learnign data, i.e. similar
# performance for different learning datasets.

vbe <- NULL
vaie <- NULL

for(j in 1:100){

  be <- NULL
  aie <- NULL
  
  for(i in 1:100){
    sigma <- matrix(data = 0, ncol=n)
    sigma <- diag(runif(n,1,2))
    x <- rmvnorm(n=1, mean=rep(0,n), sigma=sigma)
    
    be <- c(be,mean(x)^2) # bagging error
    aie <- c(aie,mean(x^2)) # average individual error
  }

  vbe <- c(vbe,var(be)) # variance of bagging error variance
  vaie <- c(vaie,var(aie)) # average individual error variance
}

hvbe <- hist(vbe)
hvaie <- hist(vaie)
plot(hvbe, col=rgb(0,0,1,1/4), xlim=c(0,1)) # bagging error variance
plot(hvaie, col=rgb(1,0,0,1/4), xlim=c(0,1), add=T) # average individual error variance

######## For both 732A99 and TDDE01

# Question: Try to predict Var from Sin.

# Answer: Impossible. You are trying to predict the inverse of the sine function (a.k.a. arcsine).
# However, such an inverse function exists only if you restrict the value of the argument of
# the sine. In other words, if you don't restict the value of Var, then you can have several
# Sin values that correspond to the same Var value. So, how to know which is the right one ?

library(neuralnet)
set.seed(1234567890)

Var <- runif(50, 0, 10)
tr <- data.frame(Var, Sin=sin(Var))
plot(tr)

winit <- runif(31, -1, 1)
nn <- neuralnet(formula = Sin ~ Var, data = tr, hidden = 10, startweights = winit,
                threshold = 0.02, lifesign = "full")

# Plot of the predictions (blue dots) and the training data (red dots)

plot(tr[,1],predict(nn,tr), col="blue", cex=3)
points(tr, col = "red", cex=3)


tr2 <- data.frame(Sin=sin(Var), Var)
plot(tr2)

nn2 <- neuralnet(formula = Var ~ Sin, data = tr2, hidden = 10, startweights = winit,
                 threshold = 0.02, lifesign = "full")

# Plot of the predictions (blue dots) and training the data (red dots)

plot(tr2[,1],predict(nn2,tr2), col="blue", cex=3)
points(tr2, col = "red", cex=3)

######### Only for TDDE01

# Question: Implement a kernel model to predict the sine function. Is it wise to use the
# leave-one-out scheme to select the kernel width ?

# Answer: Yes. The leave-one-out scheme is a particular case of cross-validation, where there
# are as many folds as data points.

n <- 100

Var <- runif(n, 0, 20)
tr <- data.frame(Var, Sin=sin(Var))
plot(tr)

h <- .1
ts <- NULL

gaussian_k <- function(x, h) { # It is fine to use exp(-x**2)/h instead
  return (exp(-(x**2)/(2*h*h)))
}

for(i in 1:n){
  ws <- 0
  t <- 0
  for(j in 1:n)
    if(i != j){
      w <- gaussian_k(tr[i,1] - tr[j,1], h)
      t <- t + tr[j,2] * w
      ws <- ws + w
    }
  ts <- c(ts, t/ws)
}

# Plot of the predictions (blue dots) and the data available (red dots)

plot(tr[,1],ts, col="blue", cex=3, ylim = c(-1.1,1.1))
points(tr, col = "red", cex=3)

mean((tr[,2]-ts)^2)

######### Only for TDDE01

# Question: Explain why the following two plots differ.

# Answer: C controls regularization, i.e. the complexity or capacity of the model. It does 
# so by controling the relative penalty for missprediction, i.e. true target values outside
# the interval prediction +- epsilon (by default epsilon=0.1), a.k.a. epsilon-tube. The 
# larger the C value the larger the penalty. This implies that the larger the C value the 
# more complex the model should be in order to avoid large penalties. That is why C=1 fits 
# the training data better than C=0.1. The downside is that the model may be fitting the 
# data too well, i.e. overfitting. So, in practice, one may need some validation data (or 
# cross-validation) to choose a C value that generalizes well.

library(kernlab)

Var <- runif(50, 0, 10)
tr <- data.frame(Var, Sin=sin(Var))

svm1 <- ksvm(Sin~.,data=tr,kernel="rbfdot",kpar=list(sigma=1),C=.1)
plot(tr[,1],predict(svm1,tr), col="blue", cex=3, ylim = c(-1.1,1.1))
points(tr, col = "red", cex=3)

svm2 <- ksvm(Sin~.,data=tr,kernel="rbfdot",kpar=list(sigma=1),C=1)
plot(tr[,1],predict(svm2,tr), col="blue", cex=3, ylim = c(-1.1,1.1))
points(tr, col = "red", cex=3)
