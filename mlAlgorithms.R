library(RWeka)
library(e1071)
library(gmodels)
library(C50)
library(caret)
library(irr)
library(randomForest)

library(dplyr)
library(class)

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

# Informedness
informedness <- function(tp, fn, tn, fp){
  
  informedness <- (tp/(tp+fn)) + (tn/(tn+fp)) -1
  
  return(informedness)
}

# Markedness
markedness <- function(tp, fp, tn, fn){
  
  markedness <- (tp/(tp+fp))+(tn/(tn+fn))-1
  
  return(markedness)
}

# F-Measure - return(precision, recall, f-measure)
getMeasues <- function(predicted, expected, positive.class="1") {
  predicted <- factor(as.character(predicted), levels=unique(as.character(expected)))
  expected  <- factor(expected, levels=unique(as.character(expected)))
  cm = as.matrix(table(expected, predicted))
  
  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)
  f1 <-  ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))
  
  informedness <- c(informedness(cm[1,1],
                                 cm[1,2]+cm[1,3],
                                 cm[2,2]+cm[3,3],
                                 cm[2,1]+cm[3,1]),
                    informedness(cm[2,2],
                                 cm[2,1]+cm[2,3],
                                 cm[1,1]+cm[3,3],
                                 cm[1,2]+cm[3,2]),
                    informedness(cm[3,3],
                                 cm[3,1]+cm[3,2],
                                 cm[1,1]+cm[2,2],
                                 cm[1,3]+cm[2,3]))
  
  markedness <- c(markedness(cm[1,1],
                             cm[2,1]+cm[3,1],
                             cm[2,2]+cm[3,3],
                             cm[1,2]+cm[1,3]),
                  markedness(cm[2,2],
                             cm[1,2]+cm[3,2],
                             cm[1,1]+cm[3,3],
                             cm[2,1]+cm[2,3]),
                  markedness(cm[3,3],
                             cm[1,3]+cm[2,3],
                             cm[1,1]+cm[2,2],
                             cm[3,1]+cm[3,2]))
  
  #Assuming that F1 is zero when it's not possible compute it
  f1[is.na(f1)] <- 0
  
  #Binary F1 or Multi-class macro-averaged F1
  ifelse(nlevels(expected) == 2, f1[positive.class], mean(f1))
  
  return(c(mean(precision), mean(recall), mean(f1), mean(informedness), mean(markedness)))
}

getMeasuresBi <- function(test, pred){
  
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
                f_measure(true_positive,false_positive,false_negative),
                informedness(true_positive, false_negative, true_negative, false_positive),
                markedness(true_positive,false_positive,true_negative,false_negative))
  return(measures)
}

####################################################### ML ###################################################################

executeNaiveBayes <- function(dataset, folds, datasetClasses){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    train_classes <- datasetClasses[-x]
    test <- dataset[x, ]
    test_classes <- datasetClasses[x]
    model <- naiveBayes(train, train_classes, laplace = 1)
    pred <- predict(model, test)
    
    results <- getMeasuresBi(test = test_classes, pred = pred)
    
    return(results)
  })
  
}

executeC50 <- function(dataset, folds, datasetClasses){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    train_classes <- datasetClasses[-x]
    test <- dataset[x, ]
    test_classes <- datasetClasses[x]
    
    
    model <- C5.0(train, factor(train_classes))
    pred <- predict(model, test)
    
    results <- getMeasuresBi(test = test_classes, pred = pred)
    
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

# dataset = dataset sem a coluna de classes
executeKNN <- function(dataset, folds, datasetClasses){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    train_classes <- datasetClasses[-x]
    test <- dataset[x, ]
    test_classes <- datasetClasses[x]
    
    pred <- knn(train = train, test = test, cl = train_classes, k = 21)
    
    #results <- measures(test_classes, pred)
    results <- getMeasues(predicted = pred, expected = test_classes)
    
    return(results)
  })
}

executeLm <- function(dataset, folds, datasetClasses){
  results <- lapply(folds, function(x) {
    train <- dataset[-x, ]
    test <- dataset[x, ]
    
    # model <- lm(train$diagnosis ~ ., data = train)
    # pred <- predict(model, test)
    # 
    # pred <- ifelse(pred>mean(pred),TRUE,FALSE)
    # test$diagnosis <- as.logical(test$diagnosis)
    # 
    # results <- getMeasuresBi(test$diagnosis,pred)
    # 
    
    model <- lm(train$survived ~ ., data = train)
    pred <- predict(model, test)
    
    pred <- normalize(pred)
    
    #se a predição estiver acima da média, considera survived=TRUE
    pred <- ifelse(pred>0.6,TRUE,FALSE)
    test$survived <- as.logical(test$survived)
    
    results <- getMeasuresBi(test$survived,pred)
    
    return(results)
  })
}

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}






################################# teste ########################################################

# dsIris <- read.csv("/home/christiano/Dropbox/trabalhos/disciplina_mineracao/datasets/datasets_menores/Iris/Iris.csv")
# 
# #retira o identificador
# dsIris <- dsIris[-1]
# #normaliza os dados
# dsIris_n <- as.data.frame(lapply(dsIris[1:4], normalize))
# # separa as classes
# classesList <- dsIris$Species
# 
# set.seed(3)
# folds <- createFolds(dsIris$Species, k =5)
# 
# knn_efetividade_by_fold <- executeKNN(dataset=dsIris_n, folds = folds, datasetClasses = classesList)
# knn_efetividade_by_fold <- as.data.frame(knn_efetividade_by_fold)
# 
# knn_efetividade <- rowMeans(knn_efetividade_by_fold)
# knn_efetividade <- as.data.frame(knn_efetividade)
# rownames(knn_efetividade) <- c("precision","recall","f-measure","informedness","markedness")
# 
# knn_efetividade


# dsBreastCancer <- read.csv("/home/christiano/Dropbox/trabalhos/disciplina_mineracao/datasets/datasets_menores/breast-cancer/data.csv", stringsAsFactors = TRUE)
# 
# #retira o identificador
# dsBreastCancer <- dsBreastCancer[c(-1)]
# # separa as classes
# classesList <- dsBreastCancer$diagnosis
# #retira as classes
# dsBreastCancer <- dsBreastCancer[-1] 
# 
# classesList <- classesList=="B"
# 
# naiveb_efetividade_by_fold <- executeNaiveBayes(dataset=dsBreastCancer, folds = folds, datasetClasses = classesList)
# naiveb_efetividade_by_fold <- as.data.frame(naiveb_efetividade_by_fold)
# 
# naiveb_efetividade <- rowMeans(naiveb_efetividade_by_fold)
# naiveb_efetividade <- as.data.frame(naiveb_efetividade)
# rownames(naiveb_efetividade) <- c("precision","recall","f-measure", "informedness","markedness")
# 
# naiveb_efetividade


# dsMushroom <- read.csv("/home/christiano/Dropbox/trabalhos/disciplina_mineracao/datasets/datasets_menores/mushrooms.csv")
# #retira o o atributo veil.type, pois só tem 1 nível
# dsMushroom <- dsMushroom[-17]
# 
# # separa as classes
# classesList <- dsMushroom$class
# #retira as classes do dataset principal
# dsMushroom <- dsMushroom[-1] 
# 
# # Facilitando o calculo de efetividade: transforma e=TRUE, p=FALSE
# ## comestível=true, não comestível=false
# classesList <- classesList=="e"
# 
# #set.seed(3)
# # definição de 20 folds
# folds <- createFolds(classesList, k =20)
# 
# dectree_efetividade_by_folds <- executeC50(dataset=dsMushroom, folds = folds, datasetClasses = classesList)
# dectree_efetividade_by_folds <- as.data.frame(dectree_efetividade_by_folds)
# 
# dectree_efetividade <- rowMeans(dectree_efetividade_by_folds)
# dectree_efetividade <- as.data.frame(dectree_efetividade)
# rownames(dectree_efetividade) <- c("precision","recall","f-measure", "informedness","markedness")
# 
# dectree_efetividade

# 
# naiveb_efetividade <- rowMeans(naiveb_efetividade_by_fold)
# naiveb_efetividade <- as.data.frame(naiveb_efetividade)
# rownames(naiveb_efetividade) <- c("precision","recall","f-measure")
# 
# naiveb_efetividade


# dectree_efetividadedectree_efetividade_by_folds <- executeC50(dataset=dsBreastCancer_n, folds = folds, datasetClasses = classesList)
# dectree_efetividade_by_folds <- as.data.frame(dectree_efetividade_by_folds)
# 
# dectree_efetividade <- rowMeans(dectree_efetividade_by_folds)
# dectree_efetividade <- as.data.frame(dectree_efetividade)
# rownames(dectree_efetividade) <- c("precision","recall","f-measure")
# 
# 
# 
# dataset <- read.csv("/home/christiano/Dropbox/trabalhos/disciplina_mineracao/datasets/datasets_menores/insurance.csv")
# set.seed(3)
# folds <- createFolds(dataset$charges, k =5)
# 
# lm_efetividade_by_folds <- executeLm(dataset, folds, dataset$charges, "charges")
# lm_efetividade_by_folds <- as.data.frame(lm_efetividade_by_folds)
# 
# lm_efetividade <- rowMeans(lm_efetividade_by_folds)
# lm_efetividade <- as.data.frame(lm_efetividade)
# rownames(lm_efetividade) <- c("precision","recall","f-measure")


test <- read.csv("/home/christiano/Dropbox/trabalhos/disciplina_mineracao/datasets/datasets_menores/titanic/test.csv", stringsAsFactors = FALSE)
gender_submission <- read.csv("/home/christiano/Dropbox/trabalhos/disciplina_mineracao/datasets/datasets_menores/titanic/gender_submission.csv", stringsAsFactors = FALSE)
train <- read.csv("/home/christiano/Dropbox/trabalhos/disciplina_mineracao/datasets/datasets_menores/titanic/train.csv", stringsAsFactors = FALSE)

test$Pclass <- as.factor(test$Pclass)
test$Sex <- as.factor(test$Sex)
test$Embarked <- as.factor(test$Embarked)
train$Pclass <- as.factor(train$Pclass)
train$Sex <- as.factor(train$Sex)
train$Embarked <- as.factor(train$Embarked)

survived <- train[,1:2]
survived <- rbind(survived, gender_submission)
survived <- survived[,-1]

# retira o atributo alvo
dataset <- train[,-2]

#insere o atributo alvo no fim do dataset
dataset <- rbind(dataset,test)
dataset <- cbind(dataset,survived=survived)

#retira atributos de natureza única: passengerId, name, ticket,
dataset <- dataset[,c(-1,-3,-8)]
#atributo cabin tem muitos valores nulos. será excluído também
dataset <- dataset[,-7]

# atribui a média para atributos com células vazias
dataset[is.na(dataset$Age),]$Age <- median(dataset$Age, na.rm = TRUE)
dataset[is.na(dataset$Fare),]$Fare <- median(dataset$Fare, na.rm = TRUE)
#atribui a categoria de maior frequencia aos campos vazios em Embarked
dataset[dataset$Embarked=='',]$Embarked <- 'S'

set.seed(3)
folds <- createFolds(dataset$survived, k =5)

lm_efetividade_by_fold <- executeLm(dataset=dataset, folds = folds)
lm_efetividade_by_fold <- as.data.frame(lm_efetividade_by_fold)

lm_efetividade <- rowMeans(lm_efetividade_by_fold)
lm_efetividade <- as.data.frame(lm_efetividade)
rownames(lm_efetividade) <- c("precision","recall","f-measure","infomedness","markdness")

lm_efetividade


# dataset <- read.csv("/home/christiano/Dropbox/trabalhos/disciplina_mineracao/datasets/datasets_menores/breast-cancer/data.csv", stringsAsFactors = TRUE)
# 
# #retira o identificador
# dataset <- dataset[c(-1)]
# 
# dataset$diagnosis <- as.numeric(dataset$diagnosis=="B")
# 
# set.seed(3)
# folds <- createFolds(dataset$diagnosis, k =5)
# 
# lm_efetividade_by_fold <- executeLm(dataset=dataset, folds = folds)
# lm_efetividade_by_fold <- as.data.frame(lm_efetividade_by_fold)
# 
# lm_efetividade <- rowMeans(lm_efetividade_by_fold)
# lm_efetividade <- as.data.frame(lm_efetividade)
# rownames(lm_efetividade) <- c("precision","recall","f-measure","infomedness","markdness")
# 
# lm_efetividade


