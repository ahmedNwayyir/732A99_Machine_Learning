RNGversion("3.5.1")
setwd("E:/Workshop/Machine Learning/Block 1/Lab 3")
set.seed(1234567890)


##Assignment 1
library(geosphere)

stations <- read.csv("Data/stations.csv",fileEncoding = "Latin1")
temps    <- read.csv("Data/temps50k.csv")
st       <- merge(stations,temps,by="station_number")
st$time  <- as.POSIXct(st$time, format="%H:%M:%S")


h_distance <- 10000
h_date     <- 900
h_time     <- 36

a <- 14.826
b <- 58.4274 
station_poi <- c(a,b)

times <- c("04:00:00", "06:00:00", "08:00:00","10:00:00",
           "12:00:00" ,"14:00:00", "16:00:00","18:00:00",
           "20:00:00","22:00:00","24:00:00")
times <- as.POSIXct(times, format="%H:%M:%S")

temp  <- matrix(0,length(times),2)
colnames(temp) <- c("Summing", "Multiplying")

k_station <- function(obs, poi)
{
  dist <- abs(distHaversine(obs, poi) / 1000)
  k    <- exp(-(dist^2)/h_distance)
  return(k)
}
k1 <- k_station(st[,c("longitude","latitude")], station_poi)


#Re-arranging the date data so only months and days is considered
dates <- as.POSIXct(st$date, format="%Y-%m-%d")
mons  <- as.numeric(format(dates,"%m"))
days  <- as.numeric(format(dates,"%d"))
dates <- cbind(mons, days)

date <- "2017-11-03" 
date <- as.POSIXct(date)
mon  <- as.numeric(format(date,"%m"))
day  <- as.numeric(format(date,"%d"))
date <- cbind(mon, day)

k_date <- function(obs, poi)
{
  diff <- (abs(obs[,1] - poi[1]) * 30) + abs(obs[,2] - poi[2])
  k    <- exp(-(diff^2)/h_date)
  return(k)
}
k2 <- k_date(dates, date)


k_time <- function(obs, poi)
{
  diff <- abs(as.numeric(difftime(obs, poi, unit = "hours")))
  k    <- exp(-(diff^2)/h_time)
  return(k)
}
k3 <- matrix(0,50000,11)
for(j in 1:length(times)){
  k3[,j] <- k_time(st$time, times[j])
}


for(j in 1:length(times)){
  temp[j,1] <- sum(k1 * st$air_temperature + k2 * st$air_temperature + k3[,j] * st$air_temperature) / sum(k1 + k2 + k3[,j])
  temp[j,2] <- sum(k1 * k2 * k3[,j] * st$air_temperature) / sum(k1 * k2 * k3[,j])
}

temp <- round(temp, 1)
knitr::kable(temp)
plot (temp[,1], type ="o", xaxt = "n", xlab ="Time of day", ylab = "Temperature", main = "Summing Kernels")
axis (1, at =1:11, labels = seq (04 ,24 ,2))

plot (temp[,2], type ="o", xaxt = "n", xlab ="Time of day", ylab = "Temperature", main = "Multiplying Kernels")
axis (1, at =1:11, labels = seq (04 ,24 ,2))





##Assignment 2

#2.1 
library("kernlab")
data("spam")

n     <- dim(spam)[1]
set.seed(12345)
id    <- sample(1:n, floor(n*0.5))
train <- spam[id,]

id1   <- setdiff(1:n, id)
set.seed(12345)
id2   <- sample(id1, floor(n*0.3))
valid <- spam[id2,]

id3   <- setdiff(id1,id2)
test  <- spam[id3,]


svm <- function(data, c){
  model     <- ksvm(type ~ ., 
                    data   = train, 
                    type   = "C-svc",
                    kernel = "rbfdot", 
                    kpar   = list(sigma = 0.05), 
                    C      = c)  
  
  pred      <- predict(model, newdata = data)  
  con_mat   <- table("Predictions" = pred, "Actuals" = data$type)
  miss_rate <- 1 - sum(diag(con_mat)) / sum(con_mat)
  
  TN <- con_mat[1,1]
  TP <- con_mat[2,2]
  FN <- con_mat[1,2]
  FP <- con_mat[2,1]
  TPR <- TP / (TP + FN)
  FPR <- FP / (FP + TN)
  
  res <- cbind(miss_rate, FPR, TPR)
  colnames(res) <- c("Error", "FPR", "TPR")
  
  if(c == 0.5){
    return(knitr::kable(res, caption = "C = 0.5"))
  }
  
  if(c == 1){
    return(knitr::kable(res, caption = "C = 1"))
  }
  
  if(c == 5){
    return(knitr::kable(res, caption = "C = 5"))
  }
}
svm(train, c = 0.5)
svm(train, c = 1)
svm(train, c = 5)

svm(valid, c = 0.5)
svm(valid, c = 1)
svm(valid, c = 5)




##2.2
svm(test, c = 1)$"Missclassification Rate"


##2.3

print(  model     <- ksvm(type ~ ., 
                          data   = train, 
                          type   = "C-svc",
                          kernel = "rbfdot", 
                          kpar   = list(sigma = 0.05), 
                          C      = 1))












svm <- function(data, c){
  model     <- ksvm(type ~ ., 
                    data   = train, 
                    type   = "C-svc",
                    kernel = "rbfdot", 
                    kpar   = list(sigma = 0.05), 
                    C      = c)  
  
  pred      <- predict(model, newdata = data)  
  con_mat   <- table("Predictions" = pred, "Actuals" = data$type)
  miss_rate <- 1 - sum(diag(con_mat)) / sum(con_mat)
  
  TN <- con_mat[1,1]
  TP <- con_mat[2,2]
  FN <- con_mat[1,2]
  FP <- con_mat[2,1]
  TPR <- TP / (TP + FN)
  FPR <- FP / (FP + TN)
  
  if(c == 0.5){
    print("C = 0.5")
    result    <- list("Missclassification Rate" = miss_rate, "FPR" = FPR, "TPR" = TPR)
    return(result)
  }
  
  if(c == 1){
    print("C = 1")
    result    <- list("Missclassification Rate" = miss_rate, "FPR" = FPR, "TPR" = TPR)
    return(result)
  }
  
  if(c == 5){
    print("C = 5")
    result    <- list("Missclassification Rate" = miss_rate, "FPR" = FPR, "TPR" = TPR)
    return(result)
  }
  
  else{
    print("Wrong Entry")
  }
}


# 3. NEURAL NETWORKS
library(neuralnet)

set.seed(1234567890)
Var  <- runif(50, 0, 10)
trva <- data.frame(Var, Sin = sin(Var))
tr   <- trva[1:25,] # Training
va   <- trva[26:50,] # Validation


# Random initialization of the weights in the interval [-1, 1]
mse     <- rep(0,10) # numeric class 10 elements with 0 values
winit   <- runif(50,-1,1) # 250 points between -1 and 1
mses    <- numeric()
for(i in 1:10) {
  set.seed(1234567890)
  nn <- neuralnet(Sin ~ Var, 
                  data         = tr, 
                  hidden       = 10,
                  threshold    = i/1000, 
                  startweights = winit)
  
  pred   <- predict(nn, va) 
  mse[i] <- mean((pred - va$Sin)^2)
}
plot(mse, type = "b", pch = 19, lwd = 1, col = ifelse(mse == mse[which.min(mse)],2,1))
best <- which.min(mse) # Most appropiate value of threshold select is i = 4

nn <- neuralnet(Sin ~ Var, 
                    data         = trva, 
                    hidden       = 10, 
                    threshold    = best/1000, 
                    startweights = winit)
plot(nn) 
plot(prediction(nn)$rep1, col="Black")  # predictions (black dots)
points(trva, col = "red") # data (red dots)


