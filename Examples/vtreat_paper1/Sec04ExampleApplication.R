## ----ReadySec04, results='hide', echo=FALSE------------------------------
rm(list=ls()) # clear workspace
source('knitrOpts.R') # TAG_TO_CAUSE_REMOVAL_DURING_TANGLE


## ----LookFTData, echo=TRUE, warning=FALSE, message=FALSE-----------------
library("Amelia")
data("freetrade")
str(freetrade)


## ----LookFTData2, echo=FALSE, warning=FALSE, message=FALSE, fig.width=6, fig.height=3----
library("ggplot2")
compRsq <- function(d, x, y) {
  meany <- mean(d[[y]])
  rsq <- 1-sum((d[[y]]-d[[x]])^2)/sum((d[[y]]-meany)^2)
  rsq
}
plotFit <- function(d, x, y, title) {
  rsq <- compRsq(d, x, y)
  ggplot(data=d, aes_string(x=x, y=y)) +
    geom_abline(color='blue') +
    geom_point() +
    ggtitle(paste(title, '\n', 'R-squared:', 
                  format(rsq, scientific=FALSE, digits=3)))
}
plotFrame <- freetrade
sortFrame <- plotFrame[plotFrame$year==1989, c('country', 'gdp.pc')]
orderedLevels <- sortFrame$country[order(-sortFrame$gdp.pc)]
plotFrame$country <- factor(plotFrame$country, orderedLevels)
ggplot(data=plotFrame, aes(x=year, y=gdp.pc, color=country, linetype=country)) +
     geom_point() + geom_line()


## ----LookFTDefGoal1------------------------------------------------------
trainData <- freetrade[freetrade$year<1990, ]
testData <- freetrade[freetrade$year>=1990, ]
origVars <- c('tariff', 'polity', 'pop', 'year', 'country')


## ----ModelPast2, echo=FALSE, warning=FALSE, message=FALSE, fig.width=6, fig.height=3----
paired <- merge(freetrade[freetrade$year==1989, c('country', 'year', 'gdp.pc')], 
               freetrade[freetrade$year>=1990, c('country', 'year', 'gdp.pc')], 
               by='country', suffixes=c('.1989', '.1990.and.after'))
plotFit(paired, 'gdp.pc.1989', 'gdp.pc.1990.and.after', 
                     '1990 and later gdp as a function of 1989 gdp, grouped by country')


## ----ClearForSec03Try1, results='hide', echo=FALSE-----------------------
rm(list=setdiff(ls(), 
                c('origVars', 'trainData', 'testData', 'compRsq', 
                  'plotFit', 'addInteractions')))


## ----ModelFTTry1, warning=FALSE, message=FALSE, results='hide'-----------
library("vtreat")
treatments <- designTreatmentsN(trainData, origVars, 'gdp.pc')
scoreFrame <- treatments$scoreFrame


## ----ModelFTTry1p2, warning=FALSE, message=FALSE-------------------------
modelingVars <- unique(c('year', 
   scoreFrame$varName[(scoreFrame$sig < 1 / nrow(scoreFrame)) &
    !(scoreFrame$code %in% c('lev'))]))
print(modelingVars)
trainTreated <- prepare(treatments, trainData, 
                       pruneSig=NULL, varRestriction=modelingVars)
testTreated <- prepare(treatments, testData, 
                       pruneSig=NULL, varRestriction=modelingVars)
formula <- paste('gdp.pc', paste(modelingVars, collapse=' + '), sep=' ~ ')
print(strwrap(formula))
model <- lm(as.formula(formula), data=trainTreated)
testTreated$pred <- predict(model, newdata=testTreated)


## ----ModelFTTry1plot, echo=FALSE, warning=FALSE, message=FALSE, fig.width=6, fig.height=3----
plotFit(testTreated, 'pred', 'gdp.pc', 
        'test gdp naively reusing training data')


## ----ClearForSec03Try2, results='hide', echo=FALSE-----------------------
rm(list=setdiff(ls(), 
                c('origVars', 'trainData', 'testData', 'compRsq', 
                  'plotFit', 'addInteractions')))


## ----ModelFTTry3, warning=FALSE, message=FALSE---------------------------
timeOrderedSplitter <- function(nRows, nSplits, dframe, y) {
  years <- sort(unique(dframe$year))
  splits <- lapply(years, function(y) {
    list(train=which(dframe$year<y), 
         app=which(dframe$year>=y))
  })
  Filter(function(si) {
          (length(si$train)>0) && (length(si$app)>0)
         }, 
         splits)
}


## ----ModelFTTry3b, warning=FALSE, message=FALSE--------------------------
cfe <- mkCrossFrameNExperiment(trainData, origVars, 'gdp.pc', 
                               splitFunction=timeOrderedSplitter)
print(cfe$method)
treatments <- cfe$treatments
scoreFrame <- treatments$scoreFrame
modelingVars <- unique(c('year', 
   scoreFrame$varName[(scoreFrame$sig < 1 / nrow(scoreFrame)) &
    !(scoreFrame$code %in% c('lev'))]))
print(modelingVars)
trainTreated <- cfe$crossFrame
testTreated <- prepare(treatments, testData, 
                       pruneSig=NULL, varRestriction=modelingVars)
formula <- paste('gdp.pc', paste(modelingVars, collapse=' + '), sep=' ~ ')
print(strwrap(formula))
model <- lm(as.formula(formula), data=trainTreated)
testTreated$pred <- predict(model, newdata=testTreated)


## ----ModelFTTry3bplot, echo=FALSE, warning=FALSE, message=FALSE, fig.width=6, fig.height=3----
plotFit(testTreated, 'pred', 'gdp.pc', 
        'test gdp using time ordered training split')

