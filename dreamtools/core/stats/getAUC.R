if (!require(pROC)) {
  install.packages("pROC")
}

library(pROC)
## These functions assume that the gold standard data and the predictions
## have already been matched 

## computes AUC and partial AUC focusing on sensitivity
##
#Assume label and prediction are matched
GetScores <- function(label, prediction, sensitivityRange = c(0.8, 1)) {
  roc1 <- roc(label, prediction, direction = "<")
  AUC <- auc(roc1)[1]
  pAUCse <- auc(roc1, partial.auc = sensitivityRange, partial.auc.focus = "sensitivity", partial.auc.correct = FALSE)[1]
  SpecAtSens <- coords(roc1, sensitivityRange[1], input = "sensitivity", ret = "specificity")
  list(AUC = AUC, pAUCse = pAUCse, SpecAtSens = SpecAtSens)
}

## Run the paired bootstrap to compute the Bayes Factors.
## 
PairedBootstrap <- function(nboot, 
                            goldData, 
                            predData,  
                            sensitivityRange = c(0.8, 1)) {
  predNames <- colnames(predData)[-c(1, 2)] 
  nPreds <- length(predNames)
  AUC <- matrix(NA, nboot, nPreds)
  colnames(AUC) <- predNames
  pAUCse <- matrix(NA, nboot, nPreds)
  colnames(pAUCse) <- predNames
  SpecAtSens <- matrix(NA, nboot, nPreds)
  colnames(SpecAtSens) <- predNames
  
  ## get subjectIDs that have a positive label for at least one breast
  ids1 <- unique(goldData$subjectID[goldData$label == 1]) 
  
  ## get subjectIDs that have negative labels in both breasts
  ids0 <- setdiff(unique(goldData$subjectID), ids1)
  
  for (i in seq(nboot)) {
    cat(i, "\n")
    
    ## get a stratified bootstrap sample
    ## note that the L and R data of each subjectID is
    ## bootstrapped as a block (both are either present 
    ## or absent from the bootstrap sample)
    ids <- c(sample(ids0, replace = TRUE), sample(ids1, replace = TRUE))
    idx <- goldData$subjectID %in% ids 
    
    ## get the bootstrap replication data
    bgold <- goldData$label[idx]
    bpred <- predData[idx, -c(1, 2), drop = FALSE]
    
    ## compute AUCs and pAUCs on the bootstraped data
    for (j in seq(nPreds)) {
      aux <- GetScores(bgold, bpred[, j], sensitivityRange)
      AUC[i, j] <- aux$AUC
      pAUCse[i, j] <- aux$pAUCse
      SpecAtSens[i, j] <- aux$SpecAtSens
    }
  }
  
  list(AUC = AUC, pAUCse = pAUCse, SpecAtSens = SpecAtSens)
}

ComputeBayesFactors <- function(bestModel, competingModel) {
  delta <- bestModel - competingModel
  BF <- sum(delta > 0)/sum(delta <= 0)
  list(BF = BF, delta = delta)
}