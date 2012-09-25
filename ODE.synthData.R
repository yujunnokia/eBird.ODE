# TODO: generate synthetic data from ODE model
#
# Author: Jun Yu
# Date: Jan 18, 2012
###############################################################################

source("ODE.R")

#
# generate synthetic data from ODFPM model
#
# Args:
#   nSites: number of sites
#	visits: a vector recording number of visits per site
#	nObservers: number of observers
#	alpha: occupancy parameter
#	beta: true detection parameter for expert
#	gamma: true detection parameter for novice
#	eta: false detection parameter for novice
#	nu: expertise parameter
#
# Returns:
#   synthetic data generated from the model including detection history, occupancy 
#	covaraites, detection covariates and true occupancy status.
#
GenerateData <- function(nSites,visits,nObservers,params)  
{
	alpha <- params$alpha
	beta  <- params$beta
	gamma <- params$gamma
	eta   <- params$eta
	nu    <- params$nu
	
    nOccCovs <- length(alpha)
    nDetCovs <- length(beta)
    nExpCovs <- length(nu)
    nVisits  <- max(visits)
    
    # generate occupancy, detection, and expertise covariates
    occCovs <- rnorm(nSites * nOccCovs, mean = 0, sd = 1)
    dim(occCovs) <- c(nSites, nOccCovs)
    occCovs[,1] <- array(1, c(nSites, 1))
    
    detCovs <- rnorm(nSites * nVisits * nDetCovs, mean = 0.5, sd = 1)
    dim(detCovs) <- c(nSites, nVisits, nDetCovs)
    detCovs[,,1] <- array(1, c(nSites, nVisits, 1))
    
    expCovs <- rnorm(nObservers * nExpCovs, mean = 0, sd = 1)
    dim(expCovs) <- c(nObservers, nExpCovs)
    expCovs[,1] <- array(1, c(nObservers, 1))
    
    # generate expertise for each birder
    expertise <- array(0, c(nObservers, 1))
    expertise <- runif(nObservers) < Logistic(expCovs %*% nu)
    
    # generate true occupancy for each birder
    trueOccs <- array(0,c(nSites,1))
    trueOccs <- runif(nSites) < Logistic(occCovs %*% alpha)
    
    # associate each checklist with one birder
    observers <- round(runif(nSites * nVisits) * nObservers + 0.5)
    dim(observers) <- c(nSites, nVisits)
    
    # generate histories	
    detHists <- array(0, c(nSites, nVisits))
    for (i in 1:nSites) {
        for (t in 1:visits[i]) {
            if (expertise[observers[i,t]] == 1) {
                isTrueDetected <- runif(1) < Logistic(detCovs[i,t,] %*% beta)
                if (trueOccs[i] == 1 && isTrueDetected == 1) {
                    detHists[i,t] <- 1
                }
            } else { 
                if (trueOccs[i] == 1) {
                    isTrueDetected <- runif(1) < Logistic(detCovs[i,t,] %*% gamma)
                    if (isTrueDetected == 1) {
                        detHists[i,t] <- 1
                    } 
                } else {
                    isFalseDetected <- runif(1) < Logistic(detCovs[i,t,] %*% eta)
                    if (isFalseDetected == 1) {
                        detHists[i,t] <- 1
                    } 
                } 
            } # expert or novice
        } # t
    } # i
    
    retData <- list(detHists=detHists, observers=observers, expertise=expertise, trueOccs=trueOccs,
            occCovs=occCovs, detCovs=detCovs, expCovs=expCovs)
    return(retData)
}
