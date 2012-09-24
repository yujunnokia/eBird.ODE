# TODO: implementation of ODE model where expert won't falsely report bird's 
#		presence, but novices might. Thus there's only one component for experts
#		and two components for novices couting for true detection and false detection
#
# Author: Jun Yu
# Date: Jan 19, 2012
###############################################################################


#####################
# Utility functions #
#####################

#
# Compute logistic function
#
# Args:
#   x: input to logistic function
#
# Returns:
#   Output of logistic function
#
Logistic <- function(x) 
{
    y <- 1 / ( 1 + exp(-x))    
    return(y)
}


#
# Implement log(exp(a) + exp(b))
#
# Args:
#   a: first input argument 
#	b: second input argument
#
# Returns:
#   output log(exp(a) + exp(b))
#
LogSumExp <- function(a,b) 
{
    c <- -Inf
    if (b < a) {
        c <- a + log(1 + exp(b - a))
    } else {
        if (a == -Inf && b == -Inf) {
            c <- -Inf
        } else {
            c <- b + log(1 + exp(a - b))
        }
    }    
    return(c)
}


#
# Implement log(exp(a) - exp(b))
#
# Args:
#   a: first input argument 
#	b: second input argument
#
# Returns:
#   output log(exp(a) - exp(b))
#
LogDiffExp <- function(a,b) 
{
    c <- -Inf    
    if (b < a) {
        c <- a + log(1 - exp(b - a))
    } else if (a == b) {
        c <- -Inf
    } else {
        warning("LogDiffExp output -inf.\n")
    }  
    return(c)
}

#
# if b is 0 and c is -inf(or the other way around), then do not add to a.
# else add b * c to a
#
Add <- function(a,b,c)
{
	if (b != 0 || c != -Inf) {
		a <- a + b * c
	} 
	
	return(a)
}

#######################
# ODE model functions #
#######################

#
# E-step: estimate the probability of Z given the model parameters
#
# Args:
#   params: a vector of occupancy, detection and expertise parameters
#	Y: observation matrix of size nSites * nVisits 
#	B: a vector of observer's index
#	E: expertise of the observers
#	Xo: occupancy covariate matrix of size nSite * nOccCovs
#	Xd: detection covariate matrix of size nSite * nVisits * nDetCovs
#	Xe: expertise covariate matrix of size nObserver * nExpCovs
#	visits: visit vecter recording number of visits to each site
#
# Returns:
#   probability of expected occupancy at each site
#
ExpectationStep <- function(params,Y,B,E,Xo,Xd,Xe,visits) 
{
    nSites     <- dim(Y)[1]  # number of sites
    nVisits    <- dim(Y)[2]  # number of visits
    nOccCovs   <- dim(Xo)[2]  # number of occupancy covs
    nDetCovs   <- dim(Xd)[3]  # number of detection covs
    nExpCovs   <- dim(Xe)[2]  # number of expertise covs
    nObservers <- dim(Xe)[1]  # number of observers
    
    alpha <- params[1:nOccCovs]
    beta  <- params[(nOccCovs+1):(nOccCovs+nDetCovs)]
    gamma <- params[(nOccCovs+nDetCovs+1):(nOccCovs+2*nDetCovs)]
    eta   <- params[(nOccCovs+2*nDetCovs+1):(nOccCovs+3*nDetCovs)]
    nu    <- params[(nOccCovs+3*nDetCovs+1):(nOccCovs+3*nDetCovs+nExpCovs)]
    
    logProbOcc <- log(Logistic(Xo %*% alpha))
	logProbExTrueDet <- array(0, c(nSites, nVisits))
	logProbNoTrueDet <- array(0, c(nSites, nVisits))
	logProbNoFalseDet <- array(0, c(nSites, nVisits))
    for (t in 1:nVisits) {
		logProbExTrueDet[,t] <- log(Logistic(Xd[,t,] %*% beta))
		logProbNoTrueDet[,t] <- log(Logistic(Xd[,t,] %*% gamma))
		logProbNoFalseDet[,t] <- log(Logistic(Xd[,t,] %*% eta))
    } # t
    
    # compute probability of expected occupancy
    probExpectedOcc <- rep(0,nSites)
    for (i in 1:nSites) {
        if (logProbOcc[i] == -Inf) {
            probExpectedOcc[i] <- 0
            next
        }
        if (LogDiffExp(0,logProbOcc[i]) == -Inf) {
            probExpectedOcc[i] <- 1
            next
        }
        
        temp1 <- 0
        temp2 <- 0
        for (t in 1:visits[i]) {
            if (E[B[i,t]] == 1) {
#                if (Y[i,t] != 0 || logProbExTrueDet[i,t] != -Inf) {
#                    temp1 <- temp1 + Y[i,t] * logProbExTrueDet[i,t] 
#                } 
#                if ((1 - Y[i,t]) != 0 || LogDiffExp(0,logProbExTrueDet[i,t]) != -Inf) {
#                    temp1 <- temp1 + (1 - Y[i,t]) * LogDiffExp(0, logProbExTrueDet[i,t])
#                }
				temp1 <- Add(temp1, Y[i,t], logProbExTrueDet[i,t])
				temp1 <- Add(temp1, (1 - Y[i,t]), LogDiffExp(0,logProbExTrueDet[i,t]))
				
                if (Y[i,t] != 0) {
                    temp2 <- temp2 + -Inf
                }
            } else {
#                if (Y[i,t] != 0 || logProbNoTrueDet[i,t] != -Inf) {
#                    temp1 <- temp1 + Y[i,t] * logProbNoTrueDet[i,t] 
#                } 
#                if ((1 - Y[i,t]) != 0 || LogDiffExp(0,logProbNoTrueDet[i,t]) != -Inf) {
#                    temp1 <- temp1 + (1 - Y[i,t]) * LogDiffExp(0, logProbNoTrueDet[i,t])
#                }
				temp1 <- Add(temp1, Y[i,t], logProbNoTrueDet[i,t])
				temp1 <- Add(temp1, (1 - Y[i,t]), LogDiffExp(0,logProbNoTrueDet[i,t]))
			
#                if (Y[i,t] != 0 || logProbNoFalseDet[i,t] != -Inf) {
#                    temp2 <- temp2 + Y[i,t] * logProbNoFalseDet[i,t] 
#                } 
#                if ((1 - Y[i,t]) != 0 || LogDiffExp(0,logProbNoFalseDet[i,t]) != -Inf) {
#                    temp2 <- temp2 + (1 - Y[i,t]) * LogDiffExp(0, logProbNoFalseDet[i,t])
#                }
				temp2 <- Add(temp2, Y[i,t], logProbNoFalseDet[i,t])
				temp2 <- Add(temp2, (1 - Y[i,t]), LogDiffExp(0,logProbNoFalseDet[i,t]))
				
            }
        } # t
        probExpectedOcc[i] <- exp(logProbOcc[i] + temp1) / (exp(logProbOcc[i] + temp1) + exp(LogDiffExp(0,logProbOcc[i]) + temp2))
        
        if (is.na(probExpectedOcc[i])) {
            stop(paste("probExpectedOcc of site", i, "is NaN", sep=" "))
        }
    } # i
    
    return(probExpectedOcc)
}

#
# compute expected joint log-likelihood (EJLL)
#
# Args:
#   params: a vector of occupancy and detection parameters
#	Y: observation matrix of size nSites * nVisits 
#	B: a vector of observer's index
#	E: expertise of the observers
#	Xo: occupancy covariate matrix of size nSite * nOccCovs
#	Xd: detection covariate matrix of size nSite * nVisits * nDetCovs
#	Xe: expertise covariate matrix of size nObserver * nExpCovs
#	visits: visit vecter recording number of visits to each site
#	probExpectedOcc: probablity of expected occupancy Z_i
#	regType: regularization type
#	lambdaO: regularization parameter for occupancy component 
#	lambdaD: regularization parameter for detection component
#	lambdaE: regularization parameter for expertise component
#
# Returns:
#   expected joint log likelihood
#
ComputeEJLL <- function(params,Y,B,E,Xo,Xd,Xe,visits,probExpectedOcc,regType,lambdaO,lambdaD,lambdaE) 
{
    nSites     <- dim(Y)[1]  # number of sites
    nVisits    <- dim(Y)[2]  # number of visits
    nOccCovs   <- dim(Xo)[2]  # number of occupancy covs
    nDetCovs   <- dim(Xd)[3]  # number of detection covs
    nExpCovs   <- dim(Xe)[2]  # number of expertise covs
    nObservers <- dim(Xe)[1]  # number of observers
    
    alpha <- params[1:nOccCovs]
    beta  <- params[(nOccCovs+1):(nOccCovs+nDetCovs)]
    gamma <- params[(nOccCovs+nDetCovs+1):(nOccCovs+2*nDetCovs)]
    eta   <- params[(nOccCovs+2*nDetCovs+1):(nOccCovs+3*nDetCovs)]
    nu    <- params[(nOccCovs+3*nDetCovs+1):(nOccCovs+3*nDetCovs+nExpCovs)]
    
    ids <- unique(c(B)) 
    ids <- ids[ids != 0]
    
	logProbOcc <- log(Logistic(Xo %*% alpha))
	logProbExp <- log(Logistic(Xe[ids,] %*% nu))
	logProbExTrueDet <- array(0, c(nSites, nVisits))
	logProbNoTrueDet <- array(0, c(nSites, nVisits))
	logProbNoFalseDet <- array(0, c(nSites, nVisits))
	for (t in 1:nVisits) {
		logProbExTrueDet[,t] <- log(Logistic(Xd[,t,] %*% beta))
		logProbNoTrueDet[,t] <- log(Logistic(Xd[,t,] %*% gamma))
		logProbNoFalseDet[,t] <- log(Logistic(Xd[,t,] %*% eta))
	} # t
    
    eJLL <- 0
    for (i in 1:nSites) {
        temp1 <- 0
        temp2 <- 0
        for (t in 1:visits[i]) {
            if (E[B[i,t]] == 1) {
#                if (Y[i,t] != 0 || logProbExTrueDet[i,t] != -Inf) {
#                    temp1 <- temp1 + Y[i,t] * logProbExTrueDet[i,t] 
#                } 
#                if ((1-Y[i,t]) != 0 || LogDiffExp(0, logProbExTrueDet[i,t]) != -Inf) {
#                    temp1 <- temp1 + (1-Y[i,t]) * LogDiffExp(0, logProbExTrueDet[i,t])
#                }
				temp1 <- Add(temp1, Y[i,t], logProbExTrueDet[i,t])
				temp1 <- Add(temp1, (1 - Y[i,t]), LogDiffExp(0,logProbExTrueDet[i,t]))
                
                if (Y[i,t] != 0) {
                    temp2 <- temp2 + -Inf
                }
            } else {
#                if (Y[i,t] != 0 || logProbNoTrueDet[i,t] != -Inf) {
#                    temp1 <- temp1 + Y[i,t]*logProbNoTrueDet[i,t] 
#                } 
#                if ((1 - Y[i,t]) != 0 || LogDiffExp(0, logProbNoTrueDet[i,t]) != -Inf) {
#                    temp1 <- temp1 + (1 - Y[i,t]) * LogDiffExp(0, logProbNoTrueDet[i,t])
#                }
				temp1 <- Add(temp1, Y[i,t], logProbNoTrueDet[i,t])
				temp1 <- Add(temp1, (1 - Y[i,t]), LogDiffExp(0,logProbNoTrueDet[i,t]))
                
#                if (Y[i,t] != 0 || logProbNoFalseDet[i,t] != -Inf) {
#                    temp2 <- temp2 + Y[i,t] * logProbNoFalseDet[i,t] 
#                } 
#                if ((1 - Y[i,t]) != 0 || LogDiffExp(0, logProbNoFalseDet[i,t]) != -Inf) {
#                    temp2 <- temp2 + (1 - Y[i,t]) * LogDiffExp(0, logProbNoFalseDet[i,t])
#                }
				temp2 <- Add(temp2, Y[i,t], logProbNoFalseDet[i,t])
				temp2 <- Add(temp2, (1 - Y[i,t]), LogDiffExp(0,logProbNoFalseDet[i,t]))
            }
        } # t
        
#        if (probExpectedOcc[i] != 0 || (logProbOcc[i] + temp1) != -Inf) {
#            eJLL <- eJLL + probExpectedOcc[i] * (logProbOcc[i] + temp1)
#        }
		eJLL <- Add(eJLL, probExpectedOcc[i], (logProbOcc[i] + temp1))
        
#        # if 1-probExpectedOcc[i] is zero, then it won't contribute to the eJLL even if temp2 is -inf
#        if ((1 - probExpectedOcc[i]) != 0 || (LogDiffExp(0, logProbOcc[i]) + temp2) != -Inf) {
#            eJLL <- eJLL + (1 - probExpectedOcc[i]) * (LogDiffExp(0, logProbOcc[i]) + temp2)
#        }
		eJLL <- Add(eJLL, (1 - probExpectedOcc[i]), (LogDiffExp(0, logProbOcc[i]) + temp2))
    } # i
    
    if (regType == 1) 
    {
        eJLL <- eJLL - lambdaO*sum(abs(alpha[2:length(alpha)])) - lambdaD*sum(abs(beta[2:(length(beta))])) - 
                lambdaD*sum(abs(gamma[2:(length(gamma))])) - lambdaD*sum(abs(eta[2:(length(eta))])) - 
                lambdaE*sum(abs(nu[2:(length(nu))]))
    }
    if (regType == 2) {
        eJLL <- eJLL - 0.5*lambdaO*sum(alpha[2:length(alpha)]^2) - 0.5*lambdaD*sum(beta[2:(length(beta))]^2) - 
                0.5*lambdaD*sum(gamma[2:(length(gamma))]^2) - 0.5*lambdaD*sum(eta[2:(length(eta))]^2) -
                0.5*lambdaE*sum(nu[2:(length(nu))]^2)
    }
    
    negEJLL <- -eJLL
    if (is.na(eJLL)) {
        stop("eJLL is na...\n")
    }
    return(negEJLL)
}

#
# compute derivatives of parameter w.r.t. EJLL
#
# Args:
#   params: a vector of occupancy and detection parameters
#	Y: observation matrix of size nSites * nVisits 
#	B: a vector of observer's index
#	E: expertise of the observers
#	Xo: occupancy covariate matrix of size nSite * nOccCovs
#	Xd: detection covariate matrix of size nSite * nVisits * nDetCovs
#	Xe: expertise covariate matrix of size nObserver * nExpCovs
#	visits: visit vecter recording number of visits to each site
#	probExpectedOcc: probablity of expected occupancy Z_i
#	regType: regularization type
#	lambdaO: regularization parameter for occupancy component 
#	lambdaD: regularization parameter for detection component
#	lambdaE: regularization parameter for expertise component
#
# Returns:
#   Derivatives of parameters
#
ComputeDerivsOfEJLL <- function(params,Y,B,E,Xo,Xd,Xe,visits,probExpectedOcc,regType,lambdaO,lambdaD,lambdaE) 
{
    nSites     <- dim(Y)[1]  # number of sites
    nVisits    <- dim(Y)[2]  # number of visits
    nOccCovs   <- dim(Xo)[2]  # number of occupancy covs
    nDetCovs   <- dim(Xd)[3]  # number of detection covs
    nExpCovs   <- dim(Xe)[2]  # number of expertise covs
    nObservers <- dim(Xe)[1]  # number of observers
    
    alpha <- params[1:nOccCovs]
    beta  <- params[(nOccCovs+1):(nOccCovs+nDetCovs)]
    gamma <- params[(nOccCovs+nDetCovs+1):(nOccCovs+2*nDetCovs)]
    eta   <- params[(nOccCovs+2*nDetCovs+1):(nOccCovs+3*nDetCovs)]
    nu    <- params[(nOccCovs+3*nDetCovs+1):(nOccCovs+3*nDetCovs+nExpCovs)]
    
    dQda <- array(0,c(nOccCovs,1))
    dQdb <- array(0,c(nDetCovs,1))
    dQdg <- array(0,c(nDetCovs,1))
    dQde <- array(0,c(nDetCovs,1))
    dQdn <- array(0,c(nExpCovs,1))
    
    probExTrueDet <- array(0, c(nSites, nVisits))
    probNoTrueDet <- array(0, c(nSites, nVisits))
    probNoFalseDet <- array(0, c(nSites, nVisits))
    for (t in 1:nVisits) {
        probExTrueDet[,t] <- Logistic(Xd[,t,] %*% beta)
        probNoTrueDet[,t] <- Logistic(Xd[,t,] %*% gamma)
        probNoFalseDet[,t] <- Logistic(Xd[,t,] %*% eta)
    } # t
    
    dQda <- t(Xo) %*% (probExpectedOcc - Logistic(Xo %*% alpha))
    
    # only count those ids with expertise covariates
    ids <- unique(c(B)) 
    ids <- ids[ids != 0]
    dQdn <- t(Xe[ids,]) %*% (E[ids,] - Logistic(Xe[ids,] %*% nu))
    
    for (i in 1:nSites) {
        for (t in 1:visits[i]) {
            if (E[B[i,t]] == 1) {
                dQdb <- dQdb + probExpectedOcc[i] * Xd[i,t,] * (Y[i,t] - probExTrueDet[i,t])
            } else {
                dQdg <- dQdg + probExpectedOcc[i] * Xd[i,t,] * (Y[i,t] - probNoTrueDet[i,t])
                dQde <- dQde + (1 - probExpectedOcc[i]) * Xd[i,t,] * (Y[i,t] - probNoFalseDet[i,t])
            }
        } # t
    } # i
    
    if (regType == 1) {
        dQda <- dQda - lambdaO * c(0, sign(alpha[2:length(alpha)]))
        dQdb <- dQdb - lambdaD * c(0, sign(beta[2:length(beta)]))
        dQdg <- dQdg - lambdaD * c(0, sign(gamma[2:length(gamma)]))
        dQde <- dQde - lambdaD * c(0, sign(eta[2:length(eta)]))
        dQdn <- dQdn - lambdaE * c(0, sign(nu[2:length(nu)]))
    }
    if (regType == 2) {
        dQda <- dQda - lambdaO * c(0, alpha[2:length(alpha)])
        dQdb <- dQdb - lambdaD * c(0, beta[2:length(beta)])
        dQdg <- dQdg - lambdaD * c(0, gamma[2:length(gamma)])
        dQde <- dQde - lambdaD * c(0, eta[2:length(eta)])
        dQdn <- dQdn - lambdaE * c(0, nu[2:length(nu)])
    }
    
    derivs <- c(-dQda,-dQdb,-dQdg,-dQde,-dQdn)
    return(derivs)
}

#
# EM
#
# Args:
#	Y: observation matrix of size nSites * nVisits 
#	B: a vector of observer's index
#	E: expertise of the observers
#	Xo: occupancy covariate matrix of size nSite * nOccCovs
#	Xd: detection covariate matrix of size nSite * nVisits * nDetCovs
#	Xe: expertise covariate matrix of size nObserver * nExpCovs
#	visits: visit vecter recording number of visits to each site
#	regType: regularization type
#	lambdaO: regularization parameter for occupancy component 
#	lambdaD: regularization parameter for detection component
#	lambdaE: regularization parameter for expertise component
#	initialParams: initial parameters
#
# Returns:
#   reestiamted parameters
#
EM <- function(Y,B,E,Xo,Xd,Xe,visits,regType,lambdaO,lambdaD,lambdaE,initialParams) 
{
    nSites     <- dim(Y)[1]  # number of sites
    nVisits    <- dim(Y)[2]  # number of visits
    nOccCovs   <- dim(Xo)[2]  # number of occupancy covs
    nDetCovs   <- dim(Xd)[3]  # number of detection covs
    nExpCovs   <- dim(Xe)[2]  # number of expertise covs
    nObservers <- dim(Xe)[1]  # number of observers
    
    params <- initialParams
    
    # set initial probExpectedOcc
    probExpectedOcc <- array(0.5,nSites)
    for (i in 1:nSites) {
        if (sum(Y[i,]) > 0) {
            probExpectedOcc[i] <- 1
        }
    } # i
    
    initialEJLL <- -ComputeEJLL(params,Y,B,E,Xo,Xd,Xe,visits,probExpectedOcc,regType,lambdaO,lambdaD,lambdaE)
    newEJLL <- initialEJLL
    newParams <- params
    diffParams <- 1.0e10
    maxIterations <- 50
    iteration <- 1
    tolerance <- 1.0e-10 #0.01
    
    while (diffParams > tolerance && iteration <= maxIterations) {
        probExpectedOcc <- ExpectationStep(params,Y,B,E,Xo,Xd,Xe,visits)
        outputs <- optim(params,ComputeEJLL,ComputeDerivsOfEJLL,Y,B,E,Xo,Xd,Xe,visits,probExpectedOcc,regType,lambdaO,lambdaD,lambdaE,method="BFGS")
        params <- outputs$par
        oldEJLL <- newEJLL
        newEJLL <- -ComputeEJLL(params,Y,B,E,Xo,Xd,Xe,visits,probExpectedOcc,regType,lambdaO,lambdaD,lambdaE)
        
        oldParams <- newParams
        newParams <- params
        diffParams <- sum((newParams-oldParams)^2) / length(newParams)
        
        cat("EM iteration: ", iteration, " EJLL improvement is ", (newEJLL - oldEJLL), 
                "params change is ", diffParams, "\n")
        iteration <- iteration + 1
    }
    
    return(params)
}

#
# random restart EM 
#
# Args:
#	Y: observation matrix of size nSites * nVisits 
#	B: a vector of observer's index
#	E: expertise of the observers
#	Xo: occupancy covariate matrix of size nSite * nOccCovs
#	Xd: detection covariate matrix of size nSite * nVisits * nDetCovs
#	Xe: expertise covariate matrix of size nObserver * nExpCovs
#	visits: visit vecter recording number of visits to each site
#	regType: regularization type
#	lambdaO: regularization parameter for occupancy component 
#	lambdaD: regularization parameter for detection component
#	lambdaE: regularization parameter for expertise component
#	nRandomRestart: number of random restarts
#
# Returns:
#   reestiamted parameters
#
RandomRestartEM <- function(Y,B,E,Xo,Xd,Xe,visits,regType=2,lambdaO=0.01,lambdaD=0.01,lambdaE=0.01,nRandomRestart=2) 
{  
	# check the input arguments
	if (dim(Y)[1] != dim(Xo)[1] || dim(Y)[1] != dim(Xd)[1] || dim(Y)[1] != length(visits)) {
		stop("Number of sites does not match.")
	}
	if (!is.element(regType,c(0,1,2))) {
		stop("regularization type is invalid. Need to be 0, 1 or 2")
	}
	if (max(B) > length(E)) {
		stop("Observer matrix B includes unknown observer index.")
	}
	
    nSites     <- dim(Y)[1]  # number of sites
    nVisits    <- dim(Y)[2]  # number of visits
    nOccCovs   <- dim(Xo)[2]  # number of occupancy covs
    nDetCovs   <- dim(Xd)[3]  # number of detection covs
    nExpCovs   <- dim(Xe)[2]  # number of expertise covs
    nObservers <- dim(Xe)[1]  # number of observers
    
    nParams <- nOccCovs + nDetCovs * 3 + nExpCovs

    # in iteration 0, we initialize parameters to be all zeros
    initialParams <- array(0, c(nParams, 1))
    params <- EM(Y,B,E,Xo,Xd,Xe,visits,regType,lambdaO,lambdaD,lambdaE,initialParams)
    probExpectedOccs <- ExpectationStep(params,Y,B,E,Xo,Xd,Xe,visits)
    alpha <- params[1:nOccCovs]
    beta  <- params[(nOccCovs+1):(nOccCovs+nDetCovs)]
    gamma <- params[(nOccCovs+nDetCovs+1):(nOccCovs+2*nDetCovs)]
    eta   <- params[(nOccCovs+2*nDetCovs+1):(nOccCovs+3*nDetCovs)]
    nu    <- params[(nOccCovs+3*nDetCovs+1):(nOccCovs+3*nDetCovs+nExpCovs)]
    bestEJLL <- -ComputeEJLL(params,Y,B,E,Xo,Xd,Xe,visits,probExpectedOccs,regType,lambdaO,lambdaD,lambdaE)
    cat("Best EJLL is",bestEJLL,"\n")
       
    # random restart
    for (i in 1:nRandomRestart) {
        cat("============================\n")
        cat("randomRestartEM Iteration ",i,"\n")
        
        done <- FALSE
        while (!done) 
        {
            initialParams <- rnorm(nParams)
            dim(initialParams) <- c(nParams,1)
            r <- try(params <- EM(Y,B,E,Xo,Xd,Xe,visits,regType,lambdaO,lambdaD,lambdaE,initialParams))
            done <- !inherits(r, "try-error")
			cat("done is",done,"\n")
        }
        
        # compute probExpectedOcc
        probExpectedOccs <- ExpectationStep(params,Y,B,E,Xo,Xd,Xe,visits)
    
        # compute expected joint log likelihood
        newEJLL <- -ComputeEJLL(params,Y,B,E,Xo,Xd,Xe,visits,probExpectedOccs,regType,lambdaO,lambdaD,lambdaE)
        cat("New ELL is",newEJLL,"\n")

        if (newEJLL > bestEJLL) {
            alpha <- params[1:nOccCovs]
            beta  <- params[(nOccCovs+1):(nOccCovs+nDetCovs)]
            gamma <- params[(nOccCovs+nDetCovs+1):(nOccCovs+2*nDetCovs)]
            eta   <- params[(nOccCovs+2*nDetCovs+1):(nOccCovs+3*nDetCovs)]
            nu    <- params[(nOccCovs+3*nDetCovs+1):(nOccCovs+3*nDetCovs+nExpCovs)]
            
            bestEJLL <- newEJLL
        }
    }
    
    print(alpha) 
    print(beta) 
    print(gamma) 
    print(eta) 
    print(nu)
    
    retData <- list(alpha=alpha, beta=beta, gamma=gamma, eta=eta, nu=nu, bestEJLL=bestEJLL)
    return(retData)
}


##############################################

#
# predict the occupancy of a new site given occupancy & detection covariates and detection history
#
# Args:
#	params: parameters of ODE model
#	Xo: occupancy covariate vector of size nOccCovs
#	Xd: detection covariate matrix of size nVisits * nDetCovs
#	Xe: expertise covariate vector of size nDetCovs
#	Y: observation vector of size nVisits 
#	nVisits: number of visits to this site. It is a scalar
#
# Returns:
#   prediction of site occupancy
#
PredictOcc <- function(params,Xo,Xd,Xe,Y,B,nVisits) 
{
    nOccCovs <- length(Xo) # number of occupancy covs
    nDetCovs <- dim(Xd)[2] # number of detection covs
    nExpCovs <- dim(Xe)[2] # number of detection covs
    
	# check input arguments
	if (length(params) != (nOccCovs+3*nDetCovs+nExpCovs)) {
		stop("The number of parameters is not consistent with number of occupancy, detetction and expertise covariates")
	}
	
    alpha <- params[1:nOccCovs]
    beta  <- params[(nOccCovs+1):(nOccCovs+nDetCovs)]
    gamma <- params[(nOccCovs+nDetCovs+1):(nOccCovs+2*nDetCovs)]
    eta   <- params[(nOccCovs+2*nDetCovs+1):(nOccCovs+3*nDetCovs)]
    nu    <- params[(nOccCovs+3*nDetCovs+1):(nOccCovs+3*nDetCovs+nExpCovs)]
    
    if (!is.null(Y)) {
        logProbOcc1 <- log(Logistic(Xo %*% alpha))
        logProbExpectedOcc0 <- LogDiffExp(0,logProbOcc1)
       
        iObservers <- B[1:nVisits]
        iUniqueObservers <- unique(iObservers)

        for (j in 1:length(iUniqueObservers)) {
            iObserver <- iUniqueObservers[j]
            iVisits <- which(iObservers == iObserver)
            
            # for each visit by birder j when site is occupied
            logProbExp1 <- log(Logistic(Xe[iObserver,] %*% nu))
            logProbExp0 <- LogDiffExp(0,logProbExp1)
            for (t in iVisits) {
#                if (Y[t] != 0 || log(Logistic(Xd[t,] %*% beta)) != -Inf) {
#                    logProbExp1 <- logProbExp1 + Y[t] * log(Logistic(Xd[t,] %*% beta))    
#                }
#                if ((1 - Y[t]) != 0 || log(1-Logistic(Xd[t,] %*% beta)) != -Inf) {
#                    logProbExp1 <- logProbExp1 + (1-Y[t]) * log(1-Logistic(Xd[t,] %*% beta))
#                }
				logProbExp1 <- Add(logProbExp1, Y[t], log(Logistic(Xd[t,] %*% beta)))
				logProbExp1 <- Add(logProbExp1, (1-Y[t]), log(1-Logistic(Xd[t,] %*% beta)))
                
#                if (Y[t] != 0 || log(Logistic(Xd[t,] %*% gamma)) != -Inf) {
#                    logProbExp0 <- logProbExp0 + Y[t] * log(Logistic(Xd[t,] %*% gamma))    
#                }
#                if ((1 - Y[t]) != 0 || log(1-Logistic(Xd[t,] %*% gamma)) != -Inf) {
#                    logProbExp0 <- logProbExp0 + (1-Y[t]) * log(1-Logistic(Xd[t,] %*% gamma))
#                }
				logProbExp0 <- Add(logProbExp0, Y[t], log(Logistic(Xd[t,] %*% gamma)))
				logProbExp0 <- Add(logProbExp0, (1-Y[t]), log(1-Logistic(Xd[t,] %*% gamma)))
            }
            logProbOcc1 <- logProbOcc1 + LogSumExp(logProbExp1,logProbExp0)
 
            # for each visit by birder j when site is NOT occupied
            logProbExp1 <- log(Logistic(Xe[iObserver,] %*% nu))
            logProbExp0 <- LogDiffExp(0,logProbExp1)
            for (t in iVisits) {
                if (Y[t] == 1) {
                    logProbExp1 <- logProbExp1 + (-Inf)
                }
                
#                if (Y[t] != 0 || log(Logistic(Xd[t,] %*% eta)) != -Inf) {
#                    logProbExp0 <- logProbExp0 + Y[t] * log(Logistic(Xd[t,] %*% eta))    
#                }
#                if ((1 - Y[t]) != 0 || log(1-Logistic(Xd[t,] %*% eta)) != -Inf) {
#                    logProbExp0 <- logProbExp0 + (1-Y[t]) * log(1-Logistic(Xd[t,] %*% eta))
#                }
				logProbExp0 <- Add(logProbExp0, Y[t], log(Logistic(Xd[t,] %*% eta)))
				logProbExp0 <- Add(logProbExp0, (1-Y[t]), log(1-Logistic(Xd[t,] %*% eta)))
            }
            logProbExpectedOcc0 <- logProbExpectedOcc0 + LogSumExp(logProbExp1,logProbExp0)
        }
        
        return(exp(logProbOcc1 - LogSumExp(logProbOcc1, logProbExpectedOcc0)))
    } else {
        return(Logistic(Xo %*% alpha))
    }
}


#
# predict the detection of a new visit at a site
#
# Args:
#	params: parameters of ODE model
#	Xo: occupancy covariate vector of size nOccCovs
#	Xd: detection covariate vector of size nDetCovs
#	Xe: expertise covariate vector of size nDetCovs
#	B: the observer ID who submitted this observation
#
# Returns:
#   prediction of detection
#
PredictDet <- function(params,Xo,Xd,Xe,B) 
{
    nOccCovs <- length(Xo) # number of occupancy covs
    nDetCovs <- length(Xd) # number of detection covs
    nExpCovs <- dim(Xe)[2] # number of expertise covs
    
	# check input arguments
	if (length(params) != (nOccCovs+3*nDetCovs+nExpCovs)) {
		stop("The number of parameters is not consistent with number of occupancy, detetction and expertise covariates")
	}
	
    alpha <- params[1:nOccCovs]
    beta  <- params[(nOccCovs+1):(nOccCovs+nDetCovs)]
    gamma <- params[(nOccCovs+nDetCovs+1):(nOccCovs+2*nDetCovs)]
    eta   <- params[(nOccCovs+2*nDetCovs+1):(nOccCovs+3*nDetCovs)]
    nu    <- params[(nOccCovs+3*nDetCovs+1):(nOccCovs+3*nDetCovs+nExpCovs)]
    
    probOcc1 <- Logistic(Xo %*% alpha)
    probExp1 <- Logistic(Xe[B,] %*% nu)
    probExTrueDet1 <- Logistic(Xd %*% beta)
    probNoTrueDet1 <- Logistic(Xd %*% gamma)
    probNoFalseDet1 <- Logistic(Xd %*% eta)
    
    probDet1 <- probOcc1 * probExp1 * probExTrueDet1 + probOcc1 * (1 - probExp1) * probNoTrueDet1 + (1 - probOcc1) * (1 - probExp1) * probNoFalseDet1
    return(probDet1)
}

#
# predict the expertise of an observer
#
#	params: parameters of ODE model
#	Xo: occupancy covariate vector of size nOccCovs
#	Xd: detection covariate matrix of size nVisits * nDetCovs
#	Xe: expertise covariate vector of size nDetCovs
#	Y: observation vector of size nVisits 
#	nVisits: number of visits to this site. It is a scalar
#
# Returns:
#   prediction of expertise
#
PredictExp <- function(params,Xo,Xd,Xe,Y,visits,B,observer) 
{
    nOccCovs <- ncol(Xo)  # number of occupancy covs
    nDetCovs <- dim(Xd)[3]  # number of detection covs
    nExpCovs <- dim(Xe)[2]  # number of expertise covs
    nSites <- nrow(Xo)  # number of sites 
    
	# check input arguments
	if (length(params) != (nOccCovs+3*nDetCovs+nExpCovs)) {
		stop("The number of parameters is not consistent with number of occupancy, detetction and expertise covariates")
	}
	
    alpha <- params[1:nOccCovs]
    beta  <- params[(nOccCovs+1):(nOccCovs+nDetCovs)]
    gamma <- params[(nOccCovs+nDetCovs+1):(nOccCovs+2*nDetCovs)]
    eta   <- params[(nOccCovs+2*nDetCovs+1):(nOccCovs+3*nDetCovs)]
    nu    <- params[(nOccCovs+3*nDetCovs+1):(nOccCovs+3*nDetCovs+nExpCovs)]
    
    if (!is.null(Y)) {
        logProbExp1 <- log(Logistic(Xe[observer,] %*% nu))
        logProbExp0 <- LogDiffExp(0,logProbExp1)       
        
        for (i in 1:nSites) {
            if (!is.element(observer,B[i,1:visits[i]])) {
                next
            }
            
            iVisits <- which(B[i,1:visits[i]] == observer)
            
            # for each visit at site i when observer is expert
            logProbOcc1 <- log(Logistic(Xo[i,] %*% alpha))
            logProbOcc0 <- LogDiffExp(0,logProbOcc1)
            for (t in iVisits) {
#                if (Y[i,t] != 0 || log(Logistic(Xd[i,t,] %*% beta)) != -Inf) {
#                    logProbOcc1 <- logProbOcc1 + Y[i,t] * log(Logistic(Xd[i,t,] %*% beta))    
#                }
#                if ((1 - Y[i,t]) != 0 || log(1-Logistic(Xd[i,t,] %*% beta)) != -Inf) {
#                    logProbOcc1 <- logProbOcc1 + (1-Y[i,t]) * log(1-Logistic(Xd[i,t,] %*% beta))
#                }
				logProbOcc1 <- Add(logProbOcc1, Y[i,t], log(Logistic(Xd[i,t,] %*% beta)))
				logProbOcc1 <- Add(logProbOcc1, (1 - Y[i,t]), log(1-Logistic(Xd[i,t,] %*% beta)))
                
                if (Y[i,t] == 1) {
                    logProbOcc0 <- logProbOcc0 + (-Inf)
                }
            }
            logProbExp1 <- logProbExp1 + LogSumExp(logProbOcc1,logProbOcc0)
            
            # for each visit at site i when observer is novice
            logProbOcc1 <- log(Logistic(Xo[i,] %*% alpha))
            logProbOcc0 <- LogDiffExp(0,logProbOcc1)
            for (t in iVisits) {
#                if (Y[i,t] != 0 || log(Logistic(Xd[i,t,] %*% gamma)) != -Inf) {
#                    logProbOcc1 <- logProbOcc1 + Y[i,t] * log(Logistic(Xd[i,t,] %*% gamma))    
#                }
#                if ((1 - Y[i,t]) != 0 || log(1-Logistic(Xd[i,t,] %*% gamma)) != -Inf) {
#                    logProbOcc1 <- logProbOcc1 + (1-Y[i,t]) * log(1-Logistic(Xd[i,t,] %*% gamma))
#                }
				logProbOcc1 <- Add(logProbOcc1, Y[i,t], log(Logistic(Xd[i,t,] %*% gamma)))
				logProbOcc1 <- Add(logProbOcc1, (1 - Y[i,t]), log(1-Logistic(Xd[i,t,] %*% gamma)))
                
#                if (Y[i,t] != 0 || log(Logistic(Xd[i,t,] %*% eta)) != -Inf) {
#                    logProbOcc0 <- logProbOcc0 + Y[i,t] * log(Logistic(Xd[i,t,] %*% eta))    
#                }
#                if ((1 - Y[i,t]) != 0 || log(1-Logistic(Xd[i,t,] %*% eta)) != -Inf) {
#                    logProbOcc0 <- logProbOcc0 + (1-Y[i,t]) * log(1-Logistic(Xd[i,t,] %*% eta))
#                }
				logProbOcc0 <- Add(logProbOcc0, Y[i,t], log(Logistic(Xd[i,t,] %*% eta)))
				logProbOcc0 <- Add(logProbOcc0, (1 - Y[i,t]), log(1-Logistic(Xd[i,t,] %*% eta)))
            }
            logProbExp0 <- logProbExp0 + LogSumExp(logProbOcc1,logProbOcc0)
        }
        
        return(exp(logProbExp1 - LogSumExp(logProbExp1, logProbExp0)))
    } else {
        return(Logistic(Xe[observer,] %*% nu))
    }
}
