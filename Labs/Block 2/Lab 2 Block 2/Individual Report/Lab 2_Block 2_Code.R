setwd('E:/Workshop/Machine Learning/Block 2/Lab 2 Block 2')
suppressWarnings(RNGversion('3.5.1'))

library(readxl)
library(ggplot2)
library(mgcv)
#library(mgcViz)


#Assignment 1. Using GAM and GLM to examine the mortality rates

## 1.1 Time series plot
flu <- read_excel("Data/influenza.xlsx")

ggplot(flu)+
  geom_line(aes(x = Time, y = Mortality, color = "Mortality")) +
  geom_line(aes(x = Time, y = Influenza, color = "Influenza")) +
  scale_color_manual("Legend", 
                     breaks = c("Mortality", "Influenza"), 
                     values = c("#00E6AC", "#FF3366"))+
  theme_minimal()



## 1.2 GAM model
w <- length(unique(flu$Week))

model <- gam(Mortality ~ Year + s(Week, k = w), 
             data   = flu, 
             family = "gaussian", 
             method = "GCV.Cp")  

#summary(model)
#par(mfrow=c(2,2))
#gam.check(model)
#vis.gam(model, type='response', plot.type='persp',
#        phi=20, theta=60,n.grid=500, border=NA)



## 1.3 Predicted vs observed mortality
pred <- predict(model)

df1 <- data.frame(Time = flu$Time, 
                  Mortality  = flu$Mortality, 
                  Prediction = pred,
                  Influenza  = flu$Influenza,
                  Residuals  = model$residuals)

ggplot(df1)+
  geom_line(aes(x = Time, y = Mortality, color = "Observed Mortality")) +
  geom_line(aes(x = Time, y = Prediction, color = "Predicted Mortality")) +
  scale_color_manual("Legend", 
                     breaks = c("Observed Mortality", "Predicted Mortality"), 
                     values = c("#FF3366", "#0071B3"))+
  theme_minimal()

summary(model)
plot(model, residuals = TRUE, cex = 2)

#ggplot(flu, aes(x = Week, y = Mortality))+
#  geom_smooth(se=F, method='gam', formula=y~s(x), color='#00CCFF')+
#  geom_point(color='#FF3366',alpha=.25)+
#  theme_minimal()



### 1.4 Prediction with Different Penalty Factors
low_model  <- gam(Mortality ~ Year + s(Week, k = w, sp=0.00001), 
                  data   = flu, 
                  family = "gaussian")
low_pred   <- predict(low_model)

summary(low_model)

high_model <- gam(Mortality ~ Year + s(Week, k = w, sp=10000), 
                  data   = flu, 
                  family = "gaussian")
high_pred  <- predict(high_model)

summary(high_model)

df2 <- data.frame(Time  = flu$Time, 
                  Mortality = flu$Mortality, 
                  pred1 = low_pred, 
                  pred2 = high_pred)

ggplot(df2, aes(x = Time))+
  geom_line(aes(y = Mortality, colour="Actual Observations"))+
  geom_line(aes(y = pred1, colour="Prediction with low penalty"))+
  geom_line(aes(y = pred2, colour="Prediction with high penalty"))+
  scale_colour_manual("Legend", 
                      breaks = c("Actual Observations", "Prediction with low penalty", "Prediction with high penalty"),
                      values = c("#FF3366","#0071B3","#00EEFF"))+
  theme_minimal()

par(mfrow = c(1,2))
plot(low_model)
plot(high_model)

# summary(low_model)
# summary(high_model)
# par(mfrow=c(2,2))
# gam.check(low_model)
# par(mfrow=c(2,2))
# gam.check(high_model)


## 1.5 Correlation between Residuals and Influenza
ggplot(df1)+
  geom_line(aes(x = Time, y = Influenza, color = "Influenza")) +
  geom_line(aes(x = Time, y = Residuals, color = "Residuals")) +
  scale_color_manual("Legend", 
                     breaks = c("Influenza", "Residuals"), 
                     values = c("#009A73", "#989898"))+
  theme_minimal()



## 1.6 Final Model 
y <- length(unique(flu$Year))
f <- length(unique(flu$Influenza))

flu_model <- gam(Mortality ~ s(Year, k = y) + s(Week, k = w) + s(Influenza, k = f), 
                 data   = flu, 
                 family = "gaussian", 
                 method = "GCV.Cp") 

flu_pred  <- predict(flu_model)

summary(flu_model)

par(mfrow=c(1,3))
plot(flu_model, residuals = TRUE, cex = 2)

# par(mfrow=c(2,2))
# gam.check(flu_model)

df3 <- data.frame(Time = flu$Time, 
                  Mortality  = flu$Mortality, 
                  Prediction = flu_pred,
                  Influenza  = flu$Influenza,
                  Residuals  = flu_model$residuals)

ggplot(df3)+
  geom_line(aes(x = Time, y = Mortality, color = "Observed Mortality")) +
  geom_line(aes(x = Time, y = Prediction, color = "Predicted Mortality")) +
  scale_color_manual("Legend", 
                     breaks = c("Observed Mortality", "Predicted Mortality"), 
                     values = c("#FF3366", "#0071B3"))+
  theme_minimal()


# vis.gam(flu_model, type='response', plot.type='persp',
#         phi=20, theta=60,n.grid=500, border=NA)









## Assignment 2. High-dimensional methods
library(pamr)
library(glmnet)
library(kernlab)

data <- read.csv2("Data/data.csv", check.names = FALSE)
data$Conference <- as.factor(data$Conference)

n     <- dim(data)[1]
set.seed(12345)
ind   <- sample(1:n, floor(n*0.7))
train <- data[ind,]
test  <- data[-ind,]
 
#train
rownames(train) <- 1:nrow(train)
x_train         <- t(train[,-4703]) # remove dependent variable
y_train         <- train[[4703]]    # vector of the dependent variable
mytrain_data    <- list(x = x_train, 
                        y = y_train, 
                        geneid    = as.character(1:nrow(x_train)), 
                        genenames = rownames(x_train))
#test 
rownames(test) <- 1:nrow(test)
x_test         <- t(test[,-4703]) 
y_test         <- test[[4703]]    


cen_model   <- pamr.train(mytrain_data)

set.seed(12345)
cvmodel     <- pamr.cv(cen_model, mytrain_data)

print(cvmodel)  
pamr.plotcv(cvmodel)

pamr.plotcen(cen_model, 
             mytrain_data, 
             threshold = cvmodel$threshold[which.min(cvmodel$error)])

features = pamr.listgenes(cen_model, 
                   mytrain_data, 
                   threshold = cvmodel$threshold[which.min(cvmodel$error)],
                   genenames = TRUE)

nrow(features)
as.matrix(features[1:10,2])

#cat(paste(colnames(data)[as.numeric(features[,1])], collapse='\n'))
# top10 <- as.matrix(colnames(data)[as.numeric(features[1:10,1])])
# top10

cen_pred <- pamr.predict(cen_model,
                         newx = x_test,
                         type = "class",
                         threshold = cvmodel$threshold[which.min(cvmodel$error)]) 

cen_mat   <- table(y_test, cen_pred)
cen_rate  <- 1 - sum(diag(cen_mat)) / sum(cen_mat)

res1 <- list("Error Rate" = cen_rate, "Features Selected" = nrow(features))





#2.2
#a
set.seed(12345)
elastic_cv <- cv.glmnet(x = t(x_train), 
                        y = y_train, 
                        family="binomial",
                        alpha = 0.5)


# par(mfrow = c(2,1))
# plot(elastic_cv)
# plot(elastic_cv$glmnet.fit)

coefs <-as.matrix(coef(elastic_cv, elastic_cv$lambda.min))
elastic_features <- length(names(coefs[coefs != 0,])) 
elastic_features

elastic_pred <- predict.cv.glmnet(elastic_cv, 
                                  newx = t(x_test), 
                                  s = elastic_cv$lambda.min,
                                  type = "class", 
                                  exact = TRUE)

elastic_mat  <- table(y_test, elastic_pred)
elastic_rate <- 1 - sum(diag(elastic_mat)) / sum(elastic_mat)

res2 <- list("Error Rate" = elastic_rate, "Features Selected" = elastic_features)



#b
set.seed(12345)
invisible(capture.output(
  svm <- ksvm(Conference ~ ., 
              data = train, 
              kernel="vanilladot",
              scaled = FALSE)))

svm@nSV
set.seed(12345)
svm_pred <- predict(svm, newdata = test)

svm_mat  <- table(y_test, elastic_pred)
svm_rate <- 1 - sum(diag(svm_mat)) / sum(svm_mat)
svm_rate
res3 <- list("Error Rate" = svm_rate, "Features Selected" = svm@nSV)

result <- cbind("NSC" = res1, "Elastic Net" = res2, "SVM" = res3)
knitr::kable(result)



#2.3
hochberg <- function(x, y, alpha) {
  p <- apply(x, 2, function(x_data){t.test(x_data ~ y, alternative = "two.sided")$p.value})

  rank     <- as.matrix(sort(p))
  l        <- length(p)
  values   <- (1:l/l) * alpha
  T_F      <- matrix(0,4702,1)
  z        <- data.frame("P-Values" = rank,"T_F" = T_F)
  
  for(i in 1:4702){
    if(rank[i] <= values[i]){
      z[i,2] <- "Rejected"
    }
    else{z[i,2] <- "Accepted"}
  }
  lowest_p <- subset(z, T_F == "Rejected")
  return(lowest_p)
}

lowest_p <- hochberg(x = data[,-4703], y = data[,4703], alpha=0.05)

cat("Top 10 features")
lowest_p[1:10,]





library(ggplot2)
ggplot() +
  ylab("P-Value") + xlab("Index") +
  geom_point(data=data.frame(x=1:length(result$features),
                             y=result$pvalues[result$mask]),
             aes(x=x, y=y), col="red") +
  geom_point(data=data.frame(x=((length(result$features) + 1):(ncol(data) -1)),
                             y=result$pvalues[!result$mask]),
             aes(x=x, y=y), col="blue")

ggplot() +
  ylab("P-Value") + xlab("Index") +
  geom_point(data=data.frame(x=1:length(result$features),
                             y=result$pvalues[result$mask]),
             aes(x=x, y=y), col="red") +
  geom_point(data=data.frame(x=((length(result$features) + 1):150),
                             y=result$pvalues[!result$mask][1:(150 - rejected)]),
             aes(x=x, y=y), col="blue")