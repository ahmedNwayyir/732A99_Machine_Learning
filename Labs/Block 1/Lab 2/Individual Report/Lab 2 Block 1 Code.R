RNGversion('3.5.1')
setwd("C:/Users/WizzCon/Desktop/Machine Learning/1. Workshop/5. Machine Learning/1. Labs/Block 1/Lab 2")

library(tree)
library(ggplot2)
library(e1071)
library(SDMTools)
library(ggrepel)
library(boot)
library(fastICA)
library(MASS)

## Assignment 1. LDA and logistic regression

#1.1
crabs <- read.csv2("Data/australian-crabs.csv" ,sep = ",",dec=".")

ggplot(crabs, aes(x = CL, y = RW))+ 
  geom_point(aes(color = sex), 
             size  = 1.5, 
             alpha = 0.8 )+
  scale_color_manual(values = c('#00CCFF', '#FF3366'))+
  labs(title  = "Australian Crabs",
       x      = "CL", 
       y      = "RW", 
       colour = "Sex")+
  theme_minimal()


#1.2
model1    <- lda(sex ~ CL+RW, data = crabs)
pred1     <- predict(model1, data = crabs)

ggplot(crabs, aes(x = CL, y = RW))+ 
  geom_point(aes(color = c("female", "male")[pred1$class]), 
             size  = 1.5, 
             alpha = 0.8 )+
  scale_color_manual(values = c('#00CCFF', '#FF3366'))+
  labs(title  = "Australian Crabs",
       x      = "CL", 
       y      = "RW", 
       colour = "Sex")+
  theme_minimal()




install.packages("devtools")
library(devtools)
install_github("fawda123/ggord")
library(ggord)
ggord(model1, crabs$sex) # doesnt work requires at least two LDA axes

library("klaR")
partimat(sex ~ CL+RW, crabs, method = "qda")

x <- seq(-6,7,0.02)
y <- seq(-6,7,0.02)
z <- as.matrix(expand.grid(x,y))
m <- length(x)
n <- length(y)

contour(x,y,matrix(pred1$class,m,n),
        levels=c(1.5,2.5,3.5),
        add=TRUE,d=FALSE,lty=2)




con_mat1   <- table("Actuals" = crabs$sex, "Predictions" = pred1$class)   
miss_rate1 <- 1 - sum(diag(con_mat1)) / sum(con_mat1)
con_mat1
miss_rate1


#1.3
model2 <- lda(sex ~ CL+RW, data = crabs, prior = c(0.1,0.9))
pred2  <- predict(model2, data = crabs)

ggplot(crabs, aes(x = CL, y = RW))+ 
  geom_point(aes(color = c("female", "male")[pred2$class]), 
             size  = 1.5, 
             alpha = 0.8 )+
  scale_color_manual(values = c('#00CCFF', '#FF3366'))+
  labs(title  = "Australian Crabs",
       x      = "CL", 
       y      = "RW", 
       colour = "Sex")+
  theme_minimal()

con_mat2   <- table("Actuals" = crabs$sex, "Predictions" = pred2$class)   
miss_rate2 <- 1 - sum(diag(con_mat2)) / sum(con_mat2)
con_mat2
miss_rate2


#1.4
model3 <- glm(sex ~ CL + RW, family = binomial, data = crabs)
pred3 <- predict(model3, data = data, type = "response")
pred3 <- as.factor(ifelse(pred3 > 0.5, "Male", "Female"))

slope <- coef(model3)[2]/(-coef(model3)[3])
intercept <- coef(model3)[1]/(-coef(model3)[3]) 

ggplot(crabs, aes(x = CL, y = RW))+ 
  geom_point(aes(color = c("female", "male")[pred3]), 
             size  = 1.5, 
             alpha = 0.8 )+
  geom_abline(intercept = intercept, 
              slope     = slope, 
              color = "black", 
              size  = 0.8,
              alpha = 0.8)+
  scale_color_manual(values = c('#00CCFF', '#FF3366'))+
  labs(title  = "Australian Crabs",
       x      = "CL", 
       y      = "RW", 
       colour = "Sex")+
  theme_minimal()

con_mat3   <- table("Actuals" = crabs$sex, "Predictions" = pred3)   
miss_rate3 <- 1 - sum(diag(con_mat3)) / sum(con_mat3)
con_mat3
miss_rate3

# x <- data.frame(RW = crabs$RW, CL = crabs$CL )
# y <- crabs$sex
# lda <- function(sex,s)
# {
#   x1      <- x[y == sex,]
#   m       <- c(mean(x1$RW), mean(x1$CL))
#   inverse <- solve(s)
#   prior   <- nrow(x1) / nrow(x) 
#   w1      <- inverse %*% m
#   b1      <- ((-1/2) %*% t(m) %*% inverse %*% m) + log(prior)
#   w1      <- as.vector(w1)
#   return(c(w1[1], w1[2], b1[1,1]))
# }
# male   <- x[y == "Male",]
# female <- x[y == "Female",]
# 
# s <- cov(male) * dim(male)[1] + cov(female) * dim(female)[1]
# s <- s / dim(x)[1]
# 
# #discriminant function coefficients
# res1 <- lda("Male",s)
# res2 <- lda("Female",s)

#decision boundary coefficients 'res'
# res <- c(-(res1[1]-res2[1]), (res2[2]-res1[2]), (res2[3]-res1[3]))

# # classification
# d <- res[1]*x[,1] + res[2]*x[,2] + res[3]
# 
# yfit <- (d>0)
# plot(x[,1], 
#      x[,2], 
#      col  = yfit + 1, 
#      xlab = "CL", 
#      ylab = "RW")

#slope and intercept
# slope <- (res[2] / res[1] ) * -1
# intercept <- res[3] /res[1] * -1

#1.3
#plot decision boundary
x <- cbind(x,sex = y)
ggplot(x, 
       aes(x = CL, y = RW))+ 
  geom_point(aes(color=sex), 
             size  = 1.5,
             alpha = 0.8)+
  scale_color_manual (values = c('#00CCFF', '#FF3366'))+
  labs(title = "LDA Descion Boundary",
       x = "CL", 
       y = "RW", 
       colour = "Sex")+
  geom_abline(slope = slope, 
              intercept = intercept)+
  theme_minimal()

#1.4
glm1 <- glm(sex ~ CL + RW,family=binomial(link="logit"), data=data)
slope1 <- -(glm1$coefficients[2] / glm1$coefficients[3] )
intercept1 <- -(glm1$coefficients[1] /glm1$coefficients[3] )
print(qplot(
  x =data$CL,
  y = data$RW,
  data = data,
  color = data$sex ,
  main="CL vs RW",
  xlab="Carapace Length", ylab = "Rear Width")
  +geom_abline(slope = slope1, intercept = intercept1,colour='purple')+ggtitle("CL Vs RW in Logistic Regression"))
cat("Decision boundary with linear regression:",slope1, "+",intercept1, "* k\n")



## Assignment 2. Analysis of credit scoring
#Step 1
data <- read.csv("Data/creditscoring.csv", header = TRUE)

n     <- dim(data)[1]
set.seed(12345)
id    <- sample(1:n, floor(n*0.5))
train <- data[id,]

id1   <- setdiff(1:n, id)
set.seed(12345)
id2   <- sample(id1, floor(n*0.25))
valid <- data[id2,]
id3   <- setdiff(id1,id2)
test  <- data[id3,]

#Step 2

D_tree <- function(data, measure){
  model     <- tree(as.factor(good_bad) ~ ., data = train, split = measure)
  fit       <- predict(model, newdata = data, type="class")
  con_mat   <- table( "Actual" = data$good_bad, "Predicted" = fit)
  miss_rate <- 1-sum(diag(con_mat))/sum(con_mat)
  
  result    <- list("Confusion Matrix" = con_mat, "Missclassification Rate" = miss_rate)
  return(result)
}

print("Training data")
D_tree(train, "deviance")

print("Testing data")
D_tree(test, "deviance")

print("Training data")
D_tree(train, "gini")

print("Testing data")
D_tree(test, "gini")

#Step 3
my_tree <- tree(as.factor(good_bad) ~ ., data = train, split = "deviance")

index      <- summary(my_tree)[4]$size
trainScore <- rep(0,index)
testScore  <- rep(0,index)

for(i in 2:index) {
  prunedTree    <- prune.tree(my_tree,best=i)
  pred          <- predict(prunedTree, newdata=valid,  type="tree")
  trainScore[i] <- deviance(prunedTree)
  testScore[i]  <- deviance(pred)
}

plot(2:index, 
     trainScore[2:index], 
     col  = "Red",
     type = "b", 
     main = "Dependence of Deviance",
     ylim = c(250,600), 
     pch  = 19, 
     cex  = 1, 
     xlab = "Number of Leaves",
     ylab = "Deviance")
points(2:index, 
       testScore[2:index], 
       col  = "Blue", 
       type = "b", 
       pch  = 19, 
       cex  = 1)


final_tree <- prune.tree(my_tree, best = 4)
final_fit  <- predict(final_tree, newdata = test, type="class")
final_mat  <- table("Actual" = test$good_bad, "Predeicted" = final_fit)
final_rate <- 1-sum(diag(final_mat))/sum(final_mat)

plot(final_tree)
text(final_tree, pretty = 0)
summary(final_tree)
list("Confusion Matrix" = final_mat, "Misclassification Rate" = final_rate)

#Step 4
bayes <- function(data){
  model     <- naiveBayes(as.factor(good_bad) ~ ., data = train)
  fit       <- predict(model, newdata = data)
  con_mat   <- table("Actual" = data$good_bad, "Predicted" = fit)
  miss_rate <- 1-sum(diag(con_mat))/sum(con_mat)
  
  result    <- list("Confusion Matrix" = con_mat, "Missclassification Rate" = miss_rate)
  return(result)
}
print("Training data")
bayes(train)
print("Testing data")
bayes(test)


#Step 5

pi          <- seq(0.05, 0.95, 0.05)

tree_fit    <- predict(final_tree, newdata = test)
tree_good   <- tree_fit[,2]
true_assign <- ifelse(test$good_bad == "good", 1, 0)

tree_TPR_FPR   <- matrix(nrow = 2, ncol = length(pi))
rownames(tree_TPR_FPR) <- c("TPR", "FPR")

for (i in 1:length(pi)){
  tree_assign <- ifelse(tree_good > pi[i], 1, 0)
  tree_mat    <- confusion.matrix(tree_assign, true_assign)
  
  tpr1 <- tree_mat[2,2]/(tree_mat[2,1] + tree_mat[2,2])
  fpr1 <- tree_mat[1,2]/(tree_mat[1,1] + tree_mat[1,2])
  
  tree_TPR_FPR[,i] <- c(tpr1,fpr1)
}

knitr::kable(round(tree_TPR_FPR,2))

#options(scipen = 999)
bayes      <- naiveBayes(good_bad ~ ., data = train)
bayes_fit  <- predict(bayes, newdata = test, type = "raw") 
bayes_good <- bayes_fit[,2]

bayes_TPR_FPR <- matrix(nrow = 2, ncol = length(pi))
rownames(bayes_TPR_FPR) <- c("TPR", "FPR")


for (i in 1:length(pi)) {
  bayes_assign <- ifelse(bayes_good > pi[i], 1, 0)
  bayes_mat    <- confusion.matrix(bayes_assign, true_assign)
  
  tpr2 <- bayes_mat[2,2]/(bayes_mat[2,1] + bayes_mat[2,2])
  fpr2 <- bayes_mat[1,2]/(bayes_mat[1,1] + bayes_mat[1,2])
  
  bayes_TPR_FPR[,i] <- c(tpr2,fpr2)
}

knitr::kable(round(bayes_TPR_FPR,2))

# ROC Optimal Tree & Naive Bayes
ggplot() + 
  geom_line(aes(x = tree_TPR_FPR[2,], y = tree_TPR_FPR[1,], col = "Optimal Tree")) + 
  geom_line(aes(x = bayes_TPR_FPR[2,], y = bayes_TPR_FPR[1,], col = "Naive Bayes")) + 
  xlab("False-Positive Rate") + 
  ylab("True-Positive Rate") +
  ggtitle("ROC")

loss_mat <- matrix(c(0,10,1,0), nrow = 2)

loss_fun <- function(data,loss_mat){
  prob        <- ifelse(data$good_bad == "good",1,0)
  
  bayes_model <- naiveBayes(as.factor(good_bad) ~ ., data = train)
  bayes_fit   <- predict(bayes_model, newdata = data, type = "raw")
  
  #To penalize the FPR, the probability of the predicted as good need to be 
  #10 times the probability of the predicted as bad to be classified as good
  bayes_fit   <- ifelse(loss_mat[1,2] * bayes_fit[,2] > loss_mat[2,1] * bayes_fit[,1],1,0)
  
  con_mat     <- table("Actual" = prob, "Predicted" = bayes_fit)
  miss_rate   <- 1-sum(diag(con_mat))/sum(con_mat)
  rownames(con_mat) <- c("Bad", "Good")
  colnames(con_mat) <- c("Bad", "Good")
  
  result    <- list("Confusion Matrix" = con_mat, "Missclassification Rate" = miss_rate)
  return(result)
}

print("Training data")
loss_fun(train,loss_mat)
print("Testing data")
loss_fun(test,loss_mat)



### Assignment 3. Uncertainty estimation

state <- read.csv2("Data/State.csv", header = TRUE)
state <- state[order(state$MET),]

ggplot(data = as.data.frame(state), aes(y = state[,1], x = state[,3]))+
  xlab("MET") + ylab("EX")+
  geom_text_repel(label = state[,8], size = 2)+
  geom_point(color = 'red') 

tree_model <- tree(EX ~ MET, 
                   data = state, 
                   control = tree.control(nobs = nrow(state), 
                                          minsize = 8))
set.seed(12345)
best_tree1 <- cv.tree(tree_model)
best_tree2 <- prune.tree(tree_model, best = 3)
summary(best_tree2)

plot(best_tree2)
text(best_tree2, pretty=1, 
     cex = 0.8, 
     xpd = TRUE)

tree_pred <- predict(best_tree2, newdata = state)

ggplot(data = as.data.frame(state), 
       aes(y = state[,1], x = state[,3])) +
  xlab("MET") + 
  ylab("EX") +
  geom_point(col = "red") +
  geom_point(x = state$MET, y = tree_pred, col = "blue")

hist(residuals(best_tree2),
     main = "Residuals Histogram",
     xlab = "Residuals")

f <- function(data, ind){
  set.seed(12345)
  sample  <- state[ind,]
  my_tree <- tree(EX ~ MET, 
                  data = sample,
                  control = tree.control(nobs = nrow(sample), minsize = 8)) 
  
  pruned_tree <- prune.tree(my_tree, best = 3) 
  
  pred    <- predict(pruned_tree, newdata = state)
  return(pred)
}

res  <- boot(state, f, R=1000)

conf <- envelope(res, level=0.95) 

ggplot(data = as.data.frame(state), 
       aes(y = state[,1], x = state[,3])) +
  xlab("MET") + 
  ylab("EX") +
  geom_point(col = "red") +
  geom_line(aes(x = state$MET, y = tree_pred), col = "blue") +
  geom_line(aes(x = state$MET, y = conf$point[1,]), col = "orange") +
  geom_line(aes(x = state$MET, y = conf$point[2,]), col = "orange")

mle <- best_tree2

rng <- function(data, mle){ 
  data1    <- data.frame(EX = data$EX, MET = data$MET) 
  n        <- length(data1$EX)
  pred     <- predict(mle, newdata = state)
  residual <- data1$EX - pred
  data1$EX <- rnorm(n, pred, sd(residual))
  return(data1)
}

f1 <- function(data){
  res      <- tree(EX ~ MET,
                   data = data, 
                   control = tree.control(nobs=nrow(state),minsize = 8))
  opt_res  <- prune.tree(res, best = 3)
  return(predict(opt_res, newdata = data))
}

f2 <- function(data){
  res      <- tree(EX ~ MET,
                   data = data, 
                   control = tree.control(nobs=nrow(state),minsize = 8))
  opt_res  <- prune.tree(res, best = 3)
  n        <- length(state$EX)
  opt_pred <- predict(opt_res, newdata = state)
  pred     <- rnorm(n,opt_pred, sd(residuals(mle)))
  return(pred)
}
set.seed(12345)
par_boot_conf <- boot(state, statistic = f1, R = 1000, mle = mle, ran.gen = rng, sim = "parametric") 
conf_interval <- envelope(par_boot_conf, level=0.95)  

set.seed(12345)
par_boot_pred <- boot(state, statistic = f2, R = 1000, mle = mle, ran.gen = rng, sim = "parametric") 
pred_interval <- envelope(par_boot_pred, level=0.95)  


ggplot(data = as.data.frame(state), 
       aes(y = state[,1], x = state[,3])) +
  xlab("MET") + 
  ylab("EX") +
  geom_point(col = "red") +
  geom_line(aes(x = state$MET, y = tree_pred), col = "blue") +
  geom_line(aes(x = state$MET, y = conf_interval$point[1,]), col = "orange") +
  geom_line(aes(x = state$MET, y = conf_interval$point[2,]), col = "orange") +
  geom_line(aes(x = state$MET, y = pred_interval$point[1,]), col = "black") +
  geom_line(aes(x = state$MET, y = pred_interval$point[2,]), col = "black")



#Assignment 4. Principal components
library("ggplot2")

data    <- read.csv2("Data/NIRspectra.csv", header = TRUE)
spectra <- data

spectra$Viscosity <- c()
comp              <- prcomp(spectra) 
lambda            <- comp$sdev^2

var               <- sprintf("%2.3f", lambda/sum(lambda)*100)

screeplot(comp, main = "Principal Components")

ggplot() +
  geom_point(aes(comp$x[,1],comp$x[,2])) +
  xlab("x1") + ylab("x2")

plot(comp$rotation[,1], 
     main="PC1 Traceplot",
     xlab = "Features",
     ylab = "Scores")
plot(comp$rotation[,2], 
     main="PC2 Traceplot",
     xlab = "Features",
     ylab = "Scores")


a   <- as.matrix(spectra)
set.seed(12345)
ica <- fastICA(a, 
               2, 
               alg.typ = "parallel", 
               fun = "logcosh", 
               alpha = 1,
               method = "R", 
               row.norm = FALSE, 
               maxit = 200, 
               tol = 0.0001, 
               verbose = TRUE) 

posterior = ica$K %*% ica$W

plot(posterior[,1], 
     main="PC1 Traceplot",
     xlab = "Features",
     ylab = "Scores")
plot(posterior[,2], 
     main="PC2 Traceplot",
     xlab = "Features",
     ylab = "Scores")

ggplot() +
  geom_point(aes(ica$S[,1],ica$S[,2])) +
  labs(x = "W1", y = "W2")