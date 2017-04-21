# Install / Load packages
install.packages("randomForest")
library("randomForest")
install.packages("e1071")
library("e1071")


# 시작 위치를 지정
TOP_DIR = "C:/Users/yourDirectory"

###### 
## Under the TOP_DIR,
##	 './rawData/train' has some subdirectories for each training case (directory name is case name)
##		in that directory, '.txt' files are selected.
##   './rawData/test' has some subdirectories for each test case (directory name is test case name)
##	 Empty './aggregatedData/' should exist.
setwd(TOP_DIR)

BASIC_FREQ_INDEX = c(12, 37, 62, 86, 111, 136, 160) #c(13, 38, 63, 88, 25, 50, 85)
BASIC_COL_NAMES = c("Freq1", "Freq3", "Freq5", "Freq7", "Freq2", "Freq4", "Freq6", "Phase1", "Phase2","Target")

FEATURE_SEL_ON = TRUE

ML_MODEL_RF = TRUE
ML_MODEL_SVM = FALSE

# Log Normalization의 이용 여부
LOGARITHM_ON = TRUE

# 위상정보 이용 여부
USE_PHASE_INFO = TRUE
if ( LOGARITHM_ON == TRUE ) {
  USE_PHASE_INFO = FALSE
}

# 스냅샷의 평균 이용 여부
TIME_AVERAGE_ON = TRUE
# 스냅샷 평균 길이
TAVERAGELENGTH = 9

setwd("./rawData/train")
caseDirList <- list.dirs(recursive = FALSE)

rm(trainRawDataSet)

for (caseDirName in caseDirList) {
  if (exists("dataset")) { rm(dataset) }
  fileList <- list.files(caseDirName, pattern = "txt$", full.names = TRUE)
  caseName = substring(caseDirName,3)
  for (file in fileList) {
    if(exists("dataset")) {
      temp_dataset <- read.table(file, header=TRUE, sep=",")[,-1]
      dataset <- rbind(dataset, temp_dataset)
      rm(temp_dataset)
    } else {
      dataset <- read.table(file, header=TRUE, sep=",")[,-1]
    }
  }
  #	dataset <- dataset[-1,]
  # trainCase <- matrix(caseName,nrow=length(dataset[,1]))
  if(exists("trainRawDataSet")) {
    trainRawDataSet <- rbind(trainRawDataSet, dataset)
    trainRawCaseSet <- rbind(trainRawCaseSet, matrix(caseName,nrow=length(dataset[,1])))
  } else {
    trainRawDataSet <- dataset
    trainRawCaseSet <- matrix(caseName,nrow=length(dataset[,1]))
  }
  
  write.csv(dataset, file = paste0("../../aggregatedData/",caseName,".csv") )
}

# 입력 snapshot들을 주파수 영역에서 특성을 파악 (column 1을 제외해서 DC성분을 제거)
trainFFTDataSet <- Mod( t( mvfft(t(trainRawDataSet))) )[,2:(length(trainRawDataSet[1,])/2)]
#phaseData <- Arg( t( mvfft(t(trainRawDataSet))) ) /pi * 180
if( USE_PHASE_INFO == TRUE ) {
  phaseData <- Arg( t( mvfft(t(trainRawDataSet))) ) /pi * 180
} else {
  phaseData <- matrix(1, nrow=dim(trainFFTDataSet)[1], ncol=1)
}

if( FEATURE_SEL_ON ) {
  if( USE_PHASE_INFO == TRUE ) {
    trainSet <- cbind(trainFFTDataSet[,BASIC_FREQ_INDEX], phaseData[,BASIC_FREQ_INDEX[1]], trainRawCaseSet)
  } else {
    trainSet <- cbind(trainFFTDataSet[,BASIC_FREQ_INDEX], trainRawCaseSet)
  }
  # trainSet <- cbind(trainFFTDataSet[,BASIC_FREQ_INDEX], rowSums(trainFFTDataSet[,c(80:84)]) trainRawCaseSet)
  #colnames(trainSet) <- BASIC_COL_NAMES
  colnames(trainSet, do.NULL = FALSE, prefix = "col")
  colnames(trainSet)[dim(trainSet)[2]] <- "Target"
} else {
  if( USE_PHASE_INFO == TRUE ) {
    trainSet <- cbind(trainFFTDataSet, phaseData[,BASIC_FREQ_INDEX[1]], trainRawCaseSet)
  } else {
    trainSet <- cbind(trainFFTDataSet, trainRawCaseSet)
  }
  colnames(trainSet, do.NULL = FALSE, prefix = "col")
  colnames(trainSet)[dim(trainSet)[2]] <- "Target"
}
rownames(trainSet) <- NULL
write.csv(trainSet, file = "../TrainSet.csv" )
table <- read.csv(file = "../TrainSet.csv", header = T)
trainSet <- table[,2:dim(table)[2]]

# 각 경우에 대한 average 값 계산
# find unique labels,
# then, gather the row indexes of each label
# 'colMeans' of those row
# save the mean values with its label
rm(averageDataLabel)
for(caseLabel in levels(trainSet[,dim(trainSet)[2]])) {
  labelPos <- which(trainSet[,dim(trainSet)[2]] == caseLabel)
  if(exists("averageDataLabel")) {
    averageDataLabel <- rbind( averageDataLabel, cbind(t(colMeans(trainSet[labelPos,(-1*dim(trainSet)[2])])), caseLabel) )
  } else {
    averageDataLabel <- cbind(t(colMeans(trainSet[labelPos,(-1*dim(trainSet)[2])])), caseLabel)
  }
}
rownames(averageDataLabel) <- NULL
write.csv(averageDataLabel, file = "../averageDataLabel.csv" )
averageDataLabel <- read.csv(file = "../averageDataLabel.csv", header = T)
averageDataLabel <- averageDataLabel[,2:dim(averageDataLabel)[2]]

####### Average of Snapshot

## train

if(TIME_AVERAGE_ON) {
  if(exists("tAveragedInput")) rm(tAveragedInput)
  for(caseLabel in levels(trainSet[,dim(trainSet)[2]])) {
    labelPos <- which(trainSet[,dim(trainSet)[2]] == caseLabel)
    for(i in 1:(length(labelPos)-TAVERAGELENGTH) ) {
      if(exists("tAveragedInput")) {
        tAveragedInput <- rbind( tAveragedInput, cbind( t( colMeans(trainSet[labelPos[i:(i+TAVERAGELENGTH-1)],(-1*dim(trainSet)[2])] )  ), caseLabel) )
      } else {
        tAveragedInput <- cbind( t( colMeans(trainSet[labelPos[i:(i+TAVERAGELENGTH-1)],(-1*dim(trainSet)[2])] ) ), caseLabel)
      }
    }
  }
  rownames(tAveragedInput) <- NULL
  #  write.csv(tAveragedInput, file = "../tAveragedInput.csv" )
  
  colnames(tAveragedInput, do.NULL = FALSE, prefix = "col")
  colnames(tAveragedInput)[dim(tAveragedInput)[2]] <- "Target"
  rownames(tAveragedInput) <- NULL
  write.csv(tAveragedInput, file = "../tAveragedInput.csv" )
  table <- read.csv(file = "../tAveragedInput.csv", header = T)
  trainSet <- table[,2:dim(table)[2]]
}


####### Logarithm

if(LOGARITHM_ON==TRUE){
  trainSet <- cbind(log10(trainSet[,-1*dim(trainSet)[2]]),trainSet[,dim(trainSet)[2]])
  colnames(trainSet, do.NULL = FALSE, prefix = "col")
  colnames(trainSet)[dim(trainSet)[2]] <- "Target"
  rownames(trainSet) <- NULL
  write.csv(trainSet, file = "../TrainSetLog.csv" )
  table <- read.csv(file = "../TrainSetLog.csv", header = T)
  trainSet <- table[,2:dim(table)[2]]
}

####### Linear Summation for pre-checking



####### modeling

# rfModel <- randomForest(Target ~ ., data = trainSet, importance = TRUE, ntree = 300, nodesize = 1)
# rfModel <- randomForest(Target ~ ., data = trainSet, importance = TRUE)
if( ML_MODEL_RF == TRUE ) {
  rfModel <- randomForest(Target ~ ., data = trainSet, importance = TRUE, ntree = 500, mtry = 7, classwt = c(1,1,3,1)) # class weight order c(0.5,1,3,1): bulb, digital, fan, none
  #imp <- sort(importance(rfModel)[,4],decreasing = TRUE)
  importance(rfModel)
} else if( ML_MODEL_SVM == TRUE ) {
  params <- tune.svm(Target ~ ., data = trainSet, gamma=10^(-6:-2), cost = 10^(-1:2))
  svmModel <- svm(Target ~ ., data = trainSet, probability = TRUE, cost = params$best.parameters[[2]], gamma = params$best.parameters[[1]])
}



####### test

setwd(TOP_DIR)
setwd("./rawData/test")

caseDirList <- list.dirs(recursive = FALSE)

rm(testRawDataSet)

for (caseDirName in caseDirList) {
  if (exists("dataset")) { rm(dataset) }
  fileList <- list.files(caseDirName, pattern = "txt$", full.names = TRUE)
  caseName = substring(caseDirName,3)
  for (file in fileList) {
    if(exists("dataset")) {
      temp_dataset <- read.table(file, header=TRUE, sep=",")[,-1]
      dataset <- rbind(dataset, temp_dataset)
      rm(temp_dataset)
    } else {
      dataset <- read.table(file, header=TRUE, sep=",")[,-1]
    }
  }
  #	dataset <- dataset[-1,]
  # trainCase <- matrix(caseName,nrow=length(dataset[,1]))
  if(exists("testRawDataSet")) {
    testRawDataSet <- rbind(testRawDataSet, dataset)
    testRawCaseSet <- rbind(testRawCaseSet, matrix(caseName,nrow=length(dataset[,1])))
  } else {
    testRawDataSet <- dataset
    testRawCaseSet <- matrix(caseName,nrow=length(dataset[,1]))
  }
  
  #write.csv(dataset, file = paste0("../../aggregatedData/",caseName,".csv") )
}

testFFTDataSet <- Mod( t( mvfft(t(testRawDataSet))) )[,2:(length(testRawDataSet[1,])/2)]
testPhaseData <- Arg( t( mvfft(t(testRawDataSet))) ) /pi * 180

if( FEATURE_SEL_ON ) {
  if( USE_PHASE_INFO == TRUE ) {
    testSet <- cbind(testFFTDataSet[,BASIC_FREQ_INDEX], testPhaseData[,BASIC_FREQ_INDEX[1]], testRawCaseSet)
  } else {
    testSet <- cbind(testFFTDataSet[,BASIC_FREQ_INDEX], testRawCaseSet)
  }
  colnames(testSet, do.NULL = FALSE, prefix = "col")
  colnames(testSet)[dim(testSet)[2]] <- "Target"
} else {
  if( USE_PHASE_INFO == TRUE ) {
    testSet <- cbind(testFFTDataSet, testPhaseData[,BASIC_FREQ_INDEX[1]], testRawCaseSet)
  } else {
    testSet <- cbind(testFFTDataSet, testRawCaseSet)
  }
  colnames(testSet, do.NULL = FALSE, prefix = "col")
  colnames(testSet)[dim(testSet)[2]] <- "Target"
}
rownames(testSet) <- NULL
write.csv(testSet, file = "../TestSet.csv" )
table <- read.csv(file = "../TestSet.csv", header = T)
testSet <- table[,2:dim(table)[2]]

# 각 경우에 대한 average 값 계산
# find unique labels,
# then, gather the row indexes of each label
# 'colMeans' of those row
# save the mean values with its label
rm(averageTestDataLabel)
for(caseLabel in levels(testSet[,dim(testSet)[2]])) {
  labelPos <- which(testSet[,dim(testSet)[2]] == caseLabel)
  if(exists("averageTestDataLabel")) {
    averageTestDataLabel <- rbind( averageTestDataLabel, cbind(t(colMeans(testSet[labelPos,(-1*dim(testSet)[2])])), caseLabel) )
  } else {
    averageTestDataLabel <- cbind(t(colMeans(testSet[labelPos,(-1*dim(testSet)[2])])), caseLabel)
  }
}
rownames(averageTestDataLabel) <- NULL
write.csv(averageTestDataLabel, file = "../averageTestDataLabel.csv" )
averageTestDataLabel <- read.csv(file = "../averageTestDataLabel.csv", header = T)
averageTestDataLabel <- averageTestDataLabel[,2:dim(averageTestDataLabel)[2]]


## time average of snapshot

if(TIME_AVERAGE_ON) {
  if(exists("tAveragedTestInput")) rm(tAveragedTestInput)
  for(caseLabel in levels(testSet[,dim(testSet)[2]])) {
    labelPos <- which(testSet[,dim(testSet)[2]] == caseLabel)
    for(i in 1:(length(labelPos)-TAVERAGELENGTH) ) {
      if(exists("tAveragedTestInput")) {
        tAveragedTestInput <- rbind( tAveragedTestInput, cbind( t( colMeans(testSet[labelPos[i:(i+TAVERAGELENGTH-1)],(-1*dim(testSet)[2])] )  ), caseLabel) )
      } else {
        tAveragedTestInput <- cbind( t( colMeans(testSet[labelPos[i:(i+TAVERAGELENGTH-1)],(-1*dim(testSet)[2])] ) ), caseLabel)
      }
    }
  }
  rownames(tAveragedTestInput) <- NULL
  #  write.csv(tAveragedInput, file = "../tAveragedInput.csv" )
  
  colnames(tAveragedTestInput, do.NULL = FALSE, prefix = "col")
  colnames(tAveragedTestInput)[dim(tAveragedTestInput)[2]] <- "Target"
  rownames(tAveragedTestInput) <- NULL
  write.csv(tAveragedTestInput, file = "../tAveragedTestInput.csv" )
  table <- read.csv(file = "../tAveragedTestInput.csv", header = T)
  testSet <- table[,2:dim(table)[2]]
}


####### Test Logarithm

if(LOGARITHM_ON==TRUE){
  testSet <- cbind(log10(testSet[,-1*dim(testSet)[2]]),testSet[,dim(testSet)[2]])
  colnames(testSet, do.NULL = FALSE, prefix = "col")
  colnames(testSet)[dim(testSet)[2]] <- "Target"
  rownames(testSet) <- NULL
  write.csv(testSet, file = "../TestSetLog.csv" )
  table <- read.csv(file = "../TestSetLog.csv", header = T)
  testSet <- table[,2:dim(table)[2]]
}


testInput <- rbind(trainSet,testSet)

# summation

# sumValue <- rowSums(testInput[,-1*(dim(testInput)[2])])
# write.csv(sumValue, file = "../powerEst.csv" )


if( ML_MODEL_RF == TRUE ) {
  predictedResult <- predict(rfModel, newdata = testInput, type = "prob")
  write.csv(predictedResult, file = "testResult.csv")
  
  matplot(predictedResult, type='l', xlab = "Test", ylab = "Prediction Probability", main = "Prediction Test Results")
  leg.txt <- colnames(predictedResult)
  legend("topright", leg.txt, col=1:5, lty = 1:5)
  
  for(caseLabel in levels(testInput[,dim(testInput)[2]])) {
    labelPos <- min(which(testInput[,dim(testInput)[2]] == caseLabel))
    text(labelPos, 1.01, caseLabel, adj=c(0,0))
  }
  
  
  
  # 각 경우에 대한 average 값 계산
  # find unique labels,
  # then, gather the row indexes of each label
  # 'colMeans' of those row
  # save the mean values with its label
  rm(averageTestDataLabel)
  for(caseLabel in levels(testInput[,dim(testInput)[2]])) {
    labelPos <- which(testInput[,dim(testInput)[2]] == caseLabel)
    if(exists("averageTestDataLabel")) {
      averageTestDataLabel <- rbind( averageTestDataLabel, cbind(t(colMeans(testInput[labelPos,(-1*dim(testInput)[2])])), caseLabel) )
    } else {
      averageTestDataLabel <- cbind(t(colMeans(testInput[labelPos,(-1*dim(testInput)[2])])), caseLabel)
    }
  }
  rownames(averageTestDataLabel) <- NULL
  write.csv(averageTestDataLabel, file = "../averageTestDataLabel.csv" )
  averageTestDataLabel <- read.csv(file = "../averageTestDataLabel.csv", header = T)
  averageTestDataLabel <- averageTestDataLabel[,2:dim(averageTestDataLabel)[2]]
  
  matplot(predictedResult, type='l', xlab = "Test", ylab = "Prediction Probability", main = "Prediction Test Results")
  leg.txt <- colnames(predictedResult)
  legend("topright", leg.txt, col=1:5, lty = 1:5)
  
  for(caseLabel in levels(testInput[,dim(testInput)[2]])) {
    labelPos <- min(which(testInput[,dim(testInput)[2]] == caseLabel))
    text(labelPos, 1.01, caseLabel, adj=c(0,0))
  }
  
} else if( ML_MODEL_SVM == TRUE ) {
  pred <- predict(svmModel, testInput, decision.values = TRUE)
  attr(pred, "decision.values")
  matplot(attr(pred, "decision.values")[,c(3,5,6)], type='l')
}
#plot(1:dim(predictedResult)[1],predictedResult[,1], type='l', col=1, xlab = "Test", ylab = "Probability")
#for(i in 2:dim(predictedResult)[2]) {
#	lines(1:dim(predictedResult)[1],predictedResult[,i], col=i)
#}






#newCase <- readline()

#testInput[1,-512] - averageDataLabel[1,-512]

### ToDo:
## Main Component (highest probability expectation)을 제거한 상황에서 expectation을 계산한 결과는 어떨까?
## 
# colSums(predictedResult)
# colMeans
