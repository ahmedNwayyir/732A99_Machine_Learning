library(tree)
library(ggplot2)
library(pls)

data <- read.csv2("../data/glass.csv")

set.seed(12345)
n <- nrow(data)

train_size <- floor(n * 0.5)
validation_size <- floor(n * 0.25)
test_size <- n - train_size - validation_size

idx <- 1:n
train_idx <- sample(x=idx, size=train_size)
validation_idx <- sample(x=idx[-train_idx], size=validation_size)
test_idx <- idx[-c(train_idx, validation_idx)]

train <- data[train_idx,]
validation <- data[validation_idx,]
test <- data[test_idx,]

## 1
sizes <- 2:8
validation_errors <- rep(0, length(sizes))
train_errors <- rep(0, length(sizes))
fit <- tree(Al ~ ., data=train)

for (size in sizes) {
  fit_pruned <- prune.tree(fit, best=size)
  validation_errors[size-1] <- mean((predict(fit_pruned, newdata=validation) - validation$Al)^2)
  train_errors[size-1] <- mean((predict(fit_pruned, newdata=train) - train$Al)^2)
}

plot_data <- data.frame(x=sizes, y1=validation_errors, y2=train_errors)

ggplot() +
  xlab("# of terminal nodes") + ylab("Mean Squarred Error") +
  geom_line(data=plot_data, aes(x=x, y=y2), color="blue") +
  geom_line(data=plot_data, aes(x=x, y=y1), color="red")

## 2
optimal_size <- which.min(validation_errors) + 1
optimal_tree <- prune.tree(fit, best=optimal_size)
test_error <- mean((predict(optimal_tree, newdata=test) - test$Al)^2)
test_error

plot(optimal_tree)
text(optimal_tree, pretty=TRUE)

## 3
set.seed(12345)
fit <- plsr(Al ~ ., data=train, validation="CV")

summary(fit)
fit$validation
fit$scores
fit$loadings

optimal_fit <- plsr(Al ~ ., data=train, ncomp=6)

## a) 3 variables
## b) 6 variables
## c) According to CV the model with 6 components is best
## d) Na Mg Si Ca Ba
## e) Y_score = z1 + z2 + z3 + z4 + z5 + z6
rowSums(optimal_fit$scores)

## f)
test_error <- mean((predict(optimal_fit, newdata=test) - test$Al)^2)
test_error

## 4