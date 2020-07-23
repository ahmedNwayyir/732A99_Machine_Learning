setwd("C:/Users/WizzCon/Desktop/Machine Learning/1. Workshop/5. Machine Learning/3. Exams/Exams with Solution/732A95 - 2018 Jan 11")
library(neuralnet)

set.seed(1234567890)
x <- runif(50,0,10)
y <- sin(x)
data <- data.frame(x,y)
  
train <- data[1:25,]
valid <- data[26:50,]

mse1   <- rep(0,10) 
init_w <- runif(22,-1,1)
for(i in 1:10) {
  nn <- neuralnet(y ~ x,
                  data = train,
                  hidden = c(10),
                  threshold = i/1000,
                  startweights = init_w,
                  lifesign = "full")
  pred    <- predict(nn, valid)
  mse1[i] <- mean((pred - valid$y)^2)
}


mse2 <- rep(0,10) 
for(i in 1:10) {
  nn <- neuralnet(y ~ x,
                  data = train,
                  hidden = c(3,3),
                  threshold = i/1000,
                  startweights = init_w,
                  lifesign = "full")
  pred    <- predict(nn, valid)
  mse2[i] <- mean((pred - valid$y)^2)
}

par(mfrow = c(1,2))
plot(mse1, type = "b", pch = 19, lwd = 1, col = ifelse(mse1 == mse1[which.min(mse1)],2,1))
plot(mse2, type = "b", pch = 19, lwd = 1, col = ifelse(mse2 == mse2[which.min(mse2)],2,1))