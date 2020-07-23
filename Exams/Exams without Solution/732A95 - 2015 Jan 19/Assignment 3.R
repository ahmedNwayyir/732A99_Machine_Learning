library(neuralnet)

data <- read.csv("../data/wine.csv")
data$class[which(data$class == 2)] <- -1

set.seed(12345)
train_idx <- sample(1:nrow(data), size=floor(nrow(data) * 0.7))
train <- data[train_idx,]
test <- data[-train_idx,]

## 3
set.seed(12345)
formula <- paste("class ~ ", paste(names(data)[-1], collapse=" + "))
fit <- neuralnet(formula=formula, data=train, hidden=0, act.fct="tanh", linear.output=FALSE)
plot(fit)

weights <- fit$weights[[1]][[1]][-1,]
weights
variables <- fit$model.list$variables[order(abs(weights), decreasing=TRUE)]
variables

## 4
train_error <- mean(sign(compute(fit, train[, -1])$net.result) != train$class)
train_error

test_error <- mean(sign(compute(fit, test[, -1])$net.result) != test$class)
test_error

## 5
set.seed(12345)
formula <- paste("class ~ ", paste(names(data)[-1], collapse=" + "))
fit <- neuralnet(formula=formula, data=train, hidden=1, act.fct="tanh", linear.output=TRUE)
plot(fit)

train_error <- mean(sign(compute(fit, train[, -1])$net.result) != train$class)
train_error

test_error <- mean(sign(compute(fit, test[, -1])$net.result) != test$class)
test_error

## 6
## 1. A tanh function
## 2. A translated tanh function
## 3. Parabola