# TODO: Learn the model parameter from the synthetic data 
#
# Author: Jun Yu
# Version: Jan 19, 2012
##################################################################

rm(list=ls())

#setwd("C:/Jun_home/workspace/eBird.ODE")
#setwd("/Users/yujunnokia/Documents/workspace/eBird.ODE")
setwd("/Users/yujunnokia/workspace/eBird.ODE")
source("ODE.R")
source("ODE.synthData.R")

library("lattice")
library("Matrix")
library("glmnet")

######################
# experiment settings
######################
nTrSites <- 500  # number of training sites
nTeSites <- 500  # number of testing sites
nVisits <- 3  # number of visits to each site
nObservers <- 50  # number of observers
nOccCovs <- 5  # number of occupancy covariates
nDetCovs <- 5  # number of detection covariates
nExpCovs <- 5  # number of expertise covariates
nParams <- nOccCovs + nDetCovs * 3 + nExpCovs + 5  # total number of paramters.
nRandomRestarts <- 1  # number of random restarts  

#################
# regularization
#################
regType <- 2 # regularization types: 0 for none, 1 for L1, 2 for L2
lambda <- lambdaO <- lambdaD <- lambdaE <- 0.01  # regularization paramters

#######################
# Set model parameters
#######################
alpha <- c(1,rnorm(nOccCovs)*3)
beta <- c(1,rnorm(nDetCovs, mean = 1.2, sd = 1)*3)
gamma <- c(1,rnorm(nDetCovs, mean = 0.6, sd = 1)*3)
eta <- c(1,rnorm(nDetCovs, mean = -1, sd = 1)*3)
nu <- c(1,rnorm(nExpCovs)*3)
covs <- rnorm(1000*nDetCovs, mean = 0.5, sd = 1)
dim(covs) <- c(1000,nDetCovs)
covs <- cbind(1,covs)
cat("mean Ex true detection prob is",mean(Logistic(covs %*% beta)),"\n",sep=" ")
cat("mean No true detection prob is",mean(Logistic(covs %*% gamma)),"\n",sep=" ")
cat("mean No false detection prob is",mean(Logistic(covs %*% eta)),"\n",sep=" ")

########################
# generate testing data
########################
teVisits <- array(nVisits, nTeSites)
teData <- GenerateData(nTeSites,teVisits,nObservers,alpha,beta,gamma,eta,nu)
teDetHists <- teData$detHists
teObservers <- teData$observers
teExpertise <- teData$expertise
teOccCovs <- teData$occCovs
teDetCovs <- teData$detCovs
teExpCovs <- teData$expCovs
teTrueOccs <- teData$trueOccs


############################
# generate training data
############################
trVisits <- array(0, c(nTrSites,1))
for (i in 1:nTrSites) {
    isMultipleVisits <- runif(1) < 0.5
    if (isMultipleVisits == TRUE) {
        trVisits[i] <- round(runif(1, min=2, max=nVisits))
    } else {
        trVisits[i] <- 1
    }
}
trData <- GenerateData(nTrSites,trVisits,nObservers,alpha,beta,gamma,eta,nu)
trDetHists <- trData$detHists
trObservers <- trData$observers
trExpertise <- trData$expertise
trOccCovs <- trData$occCovs
trDetCovs <- trData$detCovs
trExpCovs <- trData$expCovs
trTrueOccs <- trData$trueOccs

#####################
# get Bayes rates
#####################
{
    # get occupancy rate and detection rate
    teOccProb <- array(0,c(nTeSites,1))
    teExTrueDetProb <- array(0,c(nTeSites,nVisits))
    teNoTrueDetProb <- array(0,c(nTeSites,nVisits))
    teNoFalseDetProb <- array(0,c(nTeSites,nVisits))
    teExpProb <- array(0,c(nObservers,1))
    tePredExpertise <- array(0,c(nObservers,1))
    predDetHists <- array(0,c(nTeSites,nVisits))
    
    teOccProb <- Logistic(teOccCovs %*% alpha)
    teExpProb <- Logistic(teExpCovs %*% nu)
    tePredExpertise <- round(teExpProb)
    
    for (i in 1:nTeSites) {
        for (t in 1:teVisits[i]) {
            teExTrueDetProb[i,t]  <- Logistic(teDetCovs[i,t,] %*% beta)
            teNoTrueDetProb[i,t]  <- Logistic(teDetCovs[i,t,] %*% gamma)
            teNoFalseDetProb[i,t] <- Logistic(teDetCovs[i,t,] %*% eta)
            
            if (tePredExpertise[teObservers[i,t]] == 1) {
                if (round(teOccProb[i]) == 1) {
                    predDetHists[i,t] <- round(teExTrueDetProb[i,t])
                }
            } else {
                if (round(teOccProb[i]) == 1) {
                    predDetHists[i,t] <- round(teNoTrueDetProb[i,t])
                } else {
                    predDetHists[i,t] <- round(teNoFalseDetProb[i,t])				
                }
            }
        } # t
    } # i
    bayesOcc <- sum(round(teOccProb) == teTrueOccs)  / nTeSites
    bayesExp <- sum(round(teExpProb) == teExpertise) / nObservers
    bayesDet <- sum(sum(predDetHists == teDetHists)) / (sum(teVisits))
    cat("bayes occupancy rate is ",bayesOcc,"\n")
    cat("bayes expertise rate is ",bayesExp,"\n")
    cat("bayes detection rate is ",bayesDet,"\n")
}

########
# ODE
########
{
    # run ODE
    params <- RandomRestartEM(trDetHists,trObservers,trExpertise,trOccCovs,trDetCovs,
            trExpCovs,trVisits,regType,lambdaO,lambdaD,lambdaE,nRandomRestarts)
    alphaODE <- params$alpha
    betaODE <- params$beta
    gammaODE <- params$gamma
    etaODE <- params$eta
    nuODE <- params$nu
    
    # get occupancy rate and detection rate
    teOccProb <- array(0,c(nTeSites,1))
    teExTrueDetProb <- array(0,c(nTeSites,nVisits))
    teNoTrueDetProb <- array(0,c(nTeSites,nVisits))
    teNoFalseDetProb <- array(0,c(nTeSites,nVisits))
    teExpProb <- array(0,c(nObservers,1))
    tePredExpertise <- array(0,c(nObservers,1))
    predDetHists <- array(0,c(nTeSites,nVisits))
    
    teOccProb <- Logistic(teOccCovs %*% alphaODE)
    
    teExpProb <- Logistic(teExpCovs %*% nuODE)
    tePredExpertise <- round(teExpProb)
    
    for (i in 1:nTeSites) {
        for (t in 1:teVisits[i]) {
            teExTrueDetProb[i,t]  <- Logistic(teDetCovs[i,t,] %*% betaODE)
            teNoTrueDetProb[i,t]  <- Logistic(teDetCovs[i,t,] %*% gammaODE)
            teNoFalseDetProb[i,t] <- Logistic(teDetCovs[i,t,] %*% etaODE)
            
            if (tePredExpertise[teObservers[i,t]] == 1)  {
                if (round(teOccProb[i]) == 1) {
                    predDetHists[i,t] <- round(teExTrueDetProb[i,t])
                }
            } else {
                if (round(teOccProb[i]) == 1) {
                    predDetHists[i,t] <- round(teNoTrueDetProb[i,t])
                } else {
                    predDetHists[i,t] <- round(teNoFalseDetProb[i,t])				
                }
            }
        } # t
    } # i
    modelOcc <- sum(round(teOccProb) == teTrueOccs)  / nTeSites
    modelExp <- sum(round(teExpProb) == teExpertise) / nObservers
    modelDet <- sum(sum(predDetHists == teDetHists)) / (sum(teVisits))
    cat("------------------------------\n")
    cat("bayes occupancy rate is ", bayesOcc, "\n")
    cat("model occupancy rate is ", modelOcc, "\n")
    cat("bayes expertise rate is ", bayesExp, "\n")
    cat("model expertise rate is ", modelExp, "\n")
    cat("bayes detection rate is ", bayesDet, "\n")
    cat("model detection rate is ", modelDet, "\n")
    
    # predict Z on test data
    trueOccProb <- array(0,c(nTeSites,1))
    modelOccProb <- array(0,c(nTeSites,1))
    for (i in 1:nTeSites) {
        trueOccProb[i] <- PredictOcc(c(alpha,beta,gamma,eta,nu),
                teOccCovs[i,],teDetCovs[i,,],teExpCovs,teDetHists[i,],teObservers[i,],teVisits[i]) 
        modelOccProb[i] <- PredictOcc(c(alphaODE,betaODE,gammaODE,etaODE,nuODE),
                teOccCovs[i,],teDetCovs[i,,],teExpCovs,teDetHists[i,],teObservers[i,],teVisits[i]) 
    }
    trueOcc <- sum(round(trueOccProb)==teTrueOccs)/nTeSites
    predOcc <- sum(round(modelOccProb)==teTrueOccs)/nTeSites
    cat("------------------------------\n")
    cat("True occupancy prediction is ",trueOcc,"\n")
    cat("Model occupancy prediction is ",predOcc,"\n")
    
    # predict Y on test data
    trueDetHists <- array(0,c(nTeSites,nVisits))
    modelDetHists <- array(0,c(nTeSites,nVisits))
    for (i in 1:nTeSites) {
        for (t in 1:teVisits[i]) {
            trueDetHists[i,t] <- PredictDet(c(alpha,beta,gamma,eta,nu),
                    teOccCovs[i,],teDetCovs[i,t,],teExpCovs,teObservers[i,t]) 
            modelDetHists[i,t] <- PredictDet(c(alphaODE,betaODE,gammaODE,etaODE,nuODE),
                    teOccCovs[i,],teDetCovs[i,t,],teExpCovs,teObservers[i,t]) 
        }
    }
    trueDet <- sum(sum(round(trueDetHists) == teDetHists)) / (sum(teVisits))
    predDet <- sum(sum(round(modelDetHists) == teDetHists)) / (sum(teVisits))
    cat("True detection prediction is ",trueDet,"\n")
    cat("Model detection prediction is ",predDet,"\n")
    
    # predict E on test data
    trueExpProb <- array(0,c(nObservers,1))
    modelExpProb <- array(0,c(nObservers,1))
    for (j in 1:nObservers) {
        trueExpProb[j] <- PredictExp(c(alpha,beta,gamma,eta,nu),
                teOccCovs,teDetCovs,teExpCovs,teDetHists,teVisits,teObservers,j) 
        modelExpProb[j] <- PredictExp(c(alphaODE,betaODE,gammaODE,etaODE,nuODE),
                teOccCovs,teDetCovs,teExpCovs,teDetHists,teVisits,teObservers,j) 
    }
    trueExp <- sum(round(trueExpProb) == teExpertise) / nObservers
    predExp <- sum(round(modelExpProb) == teExpertise) / nObservers
    cat("True expertise prediction is ",trueExp,"\n")
    cat("Model expertise prediction is ",predExp,"\n")
    
    
    # compute MSE
    MSE <- sum(sum((alpha-alphaODE)^2) +  sum((beta-betaODE)^2) +  sum((gamma-gammaODE)^2) + 
                    sum((eta-etaODE)^2) +  sum((nu-nuODE)^2)) / nParams
    cat("------------------------------\n")
    cat("MSE is",MSE,"\n")
}

