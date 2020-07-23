library(ggplot2)
library(MASS)

data <- mtcars
data$shp <- scale(data$hp)
data$sqsec <- scale(data$qsec)
data$am <- as.factor(data$am)

## 1
ggplot() +
  geom_point(data=data, aes(x=shp, y=sqsec, color=am))

## No, the data is not linearly separable

## 2
prior <- c(1, 1) / 2
fit_eq <- lda(am ~ shp + sqsec, data=data, prior=prior)
fit_eq

prior <- as.numeric(table(data$am) / sum(table(data$am)))
fit_neq <- lda(am ~ shp + sqsec, data=data, prior=prior)
fit_neq

prediction_eq <- predict(fit_eq, data)$class
prediction_neq <- predict(fit_neq, data)$class

plot_data1 <- data.frame(x=data$shp, y=data$sqsec, color=prediction_eq, type="eq")
plot_data2 <- data.frame(x=data$shp, y=data$sqsec, color=prediction_neq, type="neq")

plot_data <- rbind(plot_data1, plot_data2)

ggplot() +
  geom_point(data=plot_data, aes(x=x, y=y, color=color)) +
  facet_grid(type ~ .)

## 3
euclidean <- function(u) {
  sqrt(sum(u^2))
}

kernel.epan <- function(u) {
  (1 - euclidean(u)^2) * as.numeric((euclidean(u) <= 1))
}

kernel.density <- function(X, Xtest, lambda) {
  apply(Xtest, 1, function(x){
    s <- 0
    
    for (i in 1:nrow(X)) {
      s <- s + kernel.epan((X[i, ] - x) / lambda)
    }
    
    s / nrow(X)
  })
}

lambda <- 0.2

idx1 <- which(data$am == 0)
X1 <- as.matrix(data.frame(data$qsec[idx1], data$hp[idx1]))
Xtest1 <- as.matrix(data.frame(data$qsec, data$hp))
density1 <- kernel.density(X1, Xtest1, lambda)

idx2 <- which(data$am == 1)
X2 <- data.frame(data$qsec[idx2], data$hp[idx2])
Xtest2 <- data.frame(data$qsec, data$hp)
density2 <- kernel.density(X2, Xtest2, lambda)

densities <- data.frame(density1, density2)
prediction <- apply(densities, 1, function(x) which.max(x))

prediction_error <- mean(as.numeric(data$am) != prediction)
prediction_error

plot_data <- data.frame(x=data$hp, y=data$qsec, color=as.factor(prediction - 1))

ggplot() +
  geom_point(data=plot_data, aes(x=x, y=y, color=color))