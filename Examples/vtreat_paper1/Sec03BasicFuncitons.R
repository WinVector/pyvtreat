## ----ReadySec03, results='hide', echo=FALSE------------------------------
rm(list=ls()) # clear workspace
source('knitrOpts.R') # TAG_TO_CAUSE_REMOVAL_DURING_TANGLE


## ----VOpsSimpleDataFrameD------------------------------------------------
d <- data.frame(
   x=c('a', 'a', 'b', 'b', NA), 
   z=c(0, 1, 2, NA, 4), 
   y=c(TRUE, TRUE, FALSE, TRUE, TRUE), 
   stringsAsFactors = FALSE)
d$yN <- as.numeric(d$y)
print(d)


## ----VTypesN1, results='hide'--------------------------------------------
library("vtreat")
treatments <- designTreatmentsN(d, c('x', 'z'), 'yN')

## ----VTypesN1s-----------------------------------------------------------
scols <- c('varName', 'sig', 'extraModelDegrees', 'origName', 'code')
print(treatments$scoreFrame[, scols])


## ----VTypesN1p-----------------------------------------------------------
dTreated <- prepare(treatments, d, pruneSig=NULL)
print(dTreated)


## ----VTypesC1, results='hide'--------------------------------------------
treatments <- designTreatmentsC(d, c('x', 'z'), 'y', TRUE)

## ----VTypesC1s-----------------------------------------------------------
print(treatments$scoreFrame[, scols])


## ----VTypesC1p-----------------------------------------------------------
dTreated <- prepare(treatments, d, pruneSig=NULL)
print(dTreated)


## ----VTypesZ1, results='hide'--------------------------------------------
treatments <- designTreatmentsZ(d, c('x', 'z'))

## ----VTypesZ1s-----------------------------------------------------------
print(treatments$scoreFrame[, scols])


## ----VTypesZ1p-----------------------------------------------------------
dTreated <- prepare(treatments, d, pruneSig=NULL)
print(dTreated)


## ----VTypesCFN1, results='hide'------------------------------------------
cfe <- mkCrossFrameNExperiment(d, c('x', 'z'), 'yN')
treatments <- cfe$treatments
dTreated <- cfe$crossFrame


## ----VTypesCFN2, results='hide'------------------------------------------
cfe <- mkCrossFrameCExperiment(d, c('x', 'z'), 'y', TRUE)
treatments <- cfe$treatments
dTreated <- cfe$crossFrame


## ----VTypesfsplitexample-------------------------------------------------
str(vtreat::oneWayHoldout(3, NULL, NULL, NULL))


## ----VTypesParellel, results='hide'--------------------------------------
ncore <- 2
parallelCluster <- parallel::makeCluster(ncore)
cfe <- mkCrossFrameNExperiment(d, c('x', 'z'), 'yN', 
   parallelCluster=parallelCluster)
parallel::stopCluster(parallelCluster)

