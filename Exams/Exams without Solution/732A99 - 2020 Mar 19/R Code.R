setwd("C:/Users/WizzCon/Desktop")
RNGversion("3.5.2")
## Assignment 1

temp <- read.csv2("Dailytemperature.csv")
phis <- matrix(0,1792,202)
temp <- cbind(temp,phis)


i <- -50
for(j in 3:103){
  for(obs in 1:1792){
    temp[obs,j]     <- sin(0.5^i * temp$Day[obs])
    temp[obs,j+101] <- cos(0.5^i * temp$Day[obs])
  }
  i <- i + 1
}

hist(temp$Temperature, breaks = 50)
library(glmnet)
lasso <- glmnet(x = as.matrix(temp[,-c(1,2)]),
                y = as.matrix(temp$Temperature),
                alpha = 1,
                family = "gaussian")

plot(lasso, xvar="lambda", label=TRUE)


set.seed(12345)
lasso_cv <- cv.glmnet(x = as.matrix(temp[,-c(1,2)]),
                      y = as.matrix(temp$Temperature),
                      alpha=1,
                      family="gaussian",
                      lambda = 0:100 * 0.001)

plot(lasso_cv)

c("Minimum Lambda" = lasso_cv$lambda.min)
c("1se Lambda" = lasso_cv$lambda.1se)


lasso_opt <- glmnet(x = as.matrix(temp[,-c(1,2)]),
                  y = as.matrix(temp$Temperature),
                  alpha = 1,
                  family = "gaussian",
                  lambda = lasso_cv$lambda.min)
lasso_opt$df
coef(lasso_opt, s = 0.052)
pred <- predict(lasso_opt, newx = as.matrix(temp[,-c(1,2)]))

mydata <- data.frame(x = temp$Day, y = temp$Temperature, yhat = pred)

plot(temp$Temperature, type = "b", col = "red")
points(pred, type = "b", col = "blue")


## Assignment 2
data(mtcars)
cars <- mtcars[,c(1,4)]
Var <- var(cars)
comps <- eigen(Var)$vectors
colnames(comps) <- c("PC1", "PC2")
c("First Component" = comps[,1])

# ggplot() +
#   geom_point(aes(comps[1,1], comps[2,1]))+
#   xlab("x1") + ylab("x2")+
#   theme_minimal()


reduced <- cbind(cars,am = mtcars$am)

ggplot(reduced, aes(x = mpg, y = hp))+
  geom_point(aes(color = as.factor(am)),
             size = 1.5,
             alpha = 0.8 )+
  scale_color_manual(values = c('#00CCFF', '#FF3366'))+
  labs(title = "Cars",
       x = "mpg",
       y = "hp",
       colour = "am")+
  theme_minimal()


library("MASS")

lda_model <- lda(am ~ mpg + hp, data = reduced)
lda_pred  <- predict(lda_model, data = reduced)

con_mat   <- table("Actuals" = reduced$am, "Predictions" = lda_pred$class)
miss_rate <- 1 - sum(diag(con_mat)) / sum(con_mat)
con_mat
miss_rate

ggplot(reduced, aes(x = mpg, y = hp))+
  geom_point(aes(color = lda_pred$class),
             size = 1.5,
             alpha = 0.8 )+
  scale_color_manual(values = c('#00CCFF', '#FF3366'))+
  labs(title = "Cars",
       x = "mpg",
       y = "hp",
       colour = "am")+
  theme_minimal()


library(mvtnorm)
m1 <- mean(cars$mpg)
m2 <- mean(cars$hp)
n  <- dim(cars)[1]
newdata <- rmvnorm(n, mean = c(m1,m2), sigma = Var)
plot(newdata)



## Assignment 3

ms <- rep(0,10)
s  <- matrix(0,10,10)
for(r in 1:10){
  for(c in 1:10){
    if(r==c){s[r,c] <- 1}
    else{
      s[r,c] <- runif(1,1,2)
    }
    s[c,r] <- s[r,c]
  }
}
s
B <- 100
b <- rmvnorm(B, mean = ms, sigma = s)





library(neuralnet)
set.seed(1234567890)
Var <- runif(50, 0, 10)
tr  <- data.frame(Var, Sin=sin(Var))
winit <- runif(31, -1, 1)
nn <- neuralnet(formula = Sin ~ Var, data = tr, hidden = 10, startweights = winit, threshold = 0.02, lifesign = "full")
plot(tr[,1],predict(nn,tr), col="blue", cex=3)
points(tr, col = "red", cex=3)


nn <- neuralnet(formula = Var ~ Sin, data = tr, hidden = 10, startweights = winit, threshold = 0.04, lifesign = "full")
plot(tr[,1],predict(nn,tr), col="blue", cex=3)
points(tr, col = "red", cex=3)





