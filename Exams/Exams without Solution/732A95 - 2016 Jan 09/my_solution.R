setwd("C:/Users/WizzCon/Desktop/Machine Learning/1. Workshop/5. Machine Learning/3. Exams/Exams without Solution/732A95 - 2016 Jan 09")
RNGversion(min(as.character(getRversion()),"3.2.3")) ## with your R-version

crx <- read.csv("crx.csv")

n <- dim(crx)[1]
set.seed(12345)
id <- sample(1:n, floor(n * 0.8))
train1 <- crx[id,]
test <- crx[-id,]

library("tree")
model <- tree(as.factor(Class) ~ ., data = train1)
plot(model)
text(model)

train2 <- train1[-2,]
model_new <- tree(as.factor(Class) ~ ., data = train2)
plot(model_new)
text(model_new, pretty = TRUE)

# summary(model)
# summary(model_new, pretty = TRUE)

index <- summary(model)[[4]]
trainScore <- rep(0,13)
testScore  <- rep(0,13)
for(k in 2:index){
  prunedTree  <- prune.tree(model, best = k)
  pred        <- predict(prunedTree, newdata = test, type = "tree")
  trainScore[k-1] <- deviance(prunedTree)
  testScore[k-1]  <- deviance(pred)
}
trainScore
testScore

plot(1:13,
     trainScore,
     col = "Red",
     type = "b",
     main = "Dependence of Deviance",
     xlim = c(1,10),
     ylim = c(0,450),
     pch = 19,
     cex = 1,
     xlab = "Training Set",
     ylab = "Deviance")
points(1:13,
       testScore,
       col = "Blue",
       type = "b",
       pch = 19,
       cex = ifelse(testScore == testScore[which.min(testScore)], 2, 1))

final_tree <- prune.tree(model, best = 4)
summary(final_tree)
# final_fit  <- predict(final_tree, newdata = test, type="class")
# con_mat    <- table("Actual" = test$Class, "Predeicted" = final_fit)
# miss_rate  <- 1-sum(diag(con_mat))/sum(con_mat)

library(mgcv)
a <- length(unique(crx$A3))
b <- length(unique(crx$A9))
mygam <- gam(Class ~  A9 + s(A3, k = a) ,
             data = crx,
             family = "binomial",
             method = "GCV.Cp")
plot(mygam)
par(mfrow = c(2,2))
gam.check(mygam)
