####################################################### Load Library ###################################################################
library(RWeka)
library(e1071)
library(gmodels)
library(C50)
library(caret)
library(irr)
library(randomForest)

####################################################### Functions ###################################################################

# Precision
precision <- function(tp, fp){
  
  precision <- tp/(tp+fp)
  
  return(precision)
}

# Recall
recall <- function(tp, fn){
  
  recall <- tp/(tp+fn)
  
  return(recall)
}

# F-measure
f_measure <- function(tp, fp, fn){
  
  f_measure <- (2*precision(tp,fp)*recall(tp,fn))/(recall(tp,fn) + precision(tp, fp))
  
  return(f_measure)
}

measures <- function(test, pred){
  
  true_positive <- 0
  true_negative <- 0
  false_positive <- 0
  false_negative <- 0
  
  for(i in 1:length(pred)){
    if(test[i] == TRUE && pred[i] == TRUE){
      true_positive <- true_positive + 1
    }else if(test[i] == FALSE && pred[i] == FALSE){
      true_negative <- true_negative + 1
    }else if(test[i] == FALSE && pred[i] == TRUE){
      false_negative <- false_negative + 1
    }else if(test[i] == TRUE && pred[i] == FALSE){
      false_positive <- false_positive + 1
    }
  }
  
  measures <- c(precision(true_positive,false_positive), 
                recall(true_positive,false_negative), 
                f_measure(true_positive,false_positive,false_negative))
  
  return(measures)
}

####################################################### Techniques ###################################################################

executeJ48 <- function(dataset, folds){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- J48(train$Affected~ ., data = train)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}

executeNaiveBayes <- function(dataset, folds){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- naiveBayes(train, train$Affected, laplace = 1)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}

executeC50 <- function(dataset, folds){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- C5.0(train, train$Affected)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}

executeSVM <- function(dataset, folds){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- svm(train$Affected~ ., data = train)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}

executeOneR <- function(dataset, folds){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- OneR(train$Affected~ ., data = train)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}

executeJRip <- function(dataset, folds){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- JRip(train$Affected~ ., data = train)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
  
}

executeRandomForest <- function(dataset, folds){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- randomForest(train$Affected~ ., data = train)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
}

executeSMO <- function(dataset, folds){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    model <- SMO(train$Affected~ ., data = train)
    pred <- predict(model, test)
    
    results <- measures(test$Affected, pred)
    
    return(results)
  })
}

####################################################### DCL Analysis ###################################################################

techniques <- c("J48", "NaiveBayes", "SVM", "oneR", "JRip", "RandomForest", "SMO")

Affecteds <- c("FE", "DCL", "GC", "II","LM", "MC", "MM", "PO","RB","SG")

# SS
#developers <- c(2, 7, 25, 28, 31, 32, 69, 86, 92, 96, 106, 107)


developers <- data.frame(c(1, 5, 6, 9, 55, 58, 60, 84, 97, 99, 101, 103),
                         c(2, 17, 18, 19, 21, 22, 27, 30, 77, 86, 93, 104),
                         c(1, 9, 13, 15, 16, 61, 62, 66, 84, 94, 102, 103),
                         c(2, 7, 21, 22, 24, 25, 28, 86, 104, 110, 111, 124),
                         c(41, 42, 43, 45, 46, 47, 49, 51, 64, 74, 81, 95),
                         c(5, 6, 10, 52, 53, 55, 58, 60, 91, 97, 99, 101),
                         c(8, 11, 39, 40, 41, 42, 43, 45, 46, 47, 74, 81),
                         c(46, 47, 49, 51, 52, 53, 64, 74, 91, 95, 105, 109),
                         c(13, 15, 16, 17, 18, 19, 30, 61, 94, 102, 111, 112),
                         c(5, 49, 51, 52, 53, 55, 56, 64, 91, 95, 99, 105))

colnames(developers) <- Affecteds

list_of_results <- list()

results <- data.frame(0,0,0, 0, 0,0,0)

dev_path <- paste(path,developers[q,j],".csv",sep="")  
dataset <- read.csv(file = "Dropbox/trabalhos/disciplina_mineracao/aulas/04-05/random_undersampling/kernel_data_balanced.csv", stringsAsFactors = FALSE)

dataset$Affected <- factor(dataset$Affected)

set.seed(3)
folds <- createFolds(dataset$Affected, k =5)

resultsJ48 <- executeJ48(dataset, folds)
partial_results <- rowMeans(as.data.frame(resultsJ48), na.rm = TRUE)

resultsNaiveBayes <- executeNaiveBayes(dataset, folds)
partial_results <- rbind(partial_results, rowMeans(as.data.frame(resultsNaiveBayes), na.rm = TRUE) ) 

resultsSVM <- executeSVM(dataset, folds)
partial_results <- rbind(partial_results, rowMeans(as.data.frame(resultsSVM), na.rm = TRUE)) 

resultsOneR <- executeOneR(dataset, folds)
partial_results <- rbind(partial_results, rowMeans(as.data.frame(resultsOneR), na.rm = TRUE)) 

resultsJRip <- executeJRip(dataset, folds)
partial_results <- rbind(partial_results, rowMeans(as.data.frame(resultsJRip), na.rm = TRUE)) 

resultsRandomForest <- executeRandomForest(dataset, folds)
partial_results <- rbind(partial_results, rowMeans(as.data.frame(resultsRandomForest), na.rm = TRUE)) 

resultsSMO <- executeSMO(dataset, folds)
partial_results <- rbind(partial_results, rowMeans(as.data.frame(resultsSMO), na.rm = TRUE)) 

rownames(partial_results) <- c("J48", "NaiveBayes", "SVM", "oneR", "JRip", "RandomForest","SMO")
colnames(partial_results) <- c("Precision", "Recall", "F-measure")

print(paste("Developer",developers[ q, j]))

print(partial_results)

results <- rbind(results, partial_results[,3])


results <- results[-1,]
rownames(results) <- developers[ ,j]
colnames(results) <- techniques
results[,] <- lapply(results,function(x){ x[is.nan(x)]<-0;return(x)})

list_of_results[[j]] <- results



print(list_of_results) 

for(Affected in 1:10){
  print(Affecteds[Affected])
  
  print(list_of_results[[Affected]])
}


results_mean <-     matrix(c(mean(list_of_results[[1]]$J48), 
                             mean(list_of_results[[1]]$NaiveBayes), 
                             mean(list_of_results[[1]]$SVM), 
                             mean(list_of_results[[1]]$oneR), 
                             mean(list_of_results[[1]]$JRip), 
                             mean(list_of_results[[1]]$RandomForest), 
                             mean(list_of_results[[1]]$SMO)), 
                           nrow = 1,
                           ncol = 7)

for(Affected in 2:10){
  results_mean <- rbind(results_mean, c(mean(list_of_results[[Affected]]$J48), 
                                        mean(list_of_results[[Affected]]$NaiveBayes), 
                                        mean(list_of_results[[Affected]]$SVM), 
                                        mean(list_of_results[[Affected]]$oneR), 
                                        mean(list_of_results[[Affected]]$JRip), 
                                        mean(list_of_results[[Affected]]$RandomForest), 
                                        mean(list_of_results[[Affected]]$SMO)))
}

results_mean


colnames(results_mean) <- techniques
rownames(results_mean) <- colnames(developers)
results_mean <- t(results_mean)
results_mean

barplot(results_mean, 
        main="Code Affecteds x Effectiveness",
        ylab="Effectiveness",
        xlab="Techniques", 
        col=c("red", "yellow", "green", "violet", "orange", "blue", "pink"), 
        ylim = c(0, 1),
        #legend = rownames(results_mean), 
        beside=TRUE)



