## ----ReadySec02, results='hide', echo=FALSE------------------------------
rm(list=ls()) # clear workspace
source('knitrOpts.R') # TAG_TO_CAUSE_REMOVAL_DURING_TANGLE


## ----msleep, tidy=TRUE---------------------------------------------------
library("ggplot2")
data("msleep")
str(msleep, width=70, strict.width='cut')


## ----isbad---------------------------------------------------------------
msleep$sleep_rem_isBAD <- is.na(msleep$sleep_rem)
msleep$sleep_rem <- ifelse(msleep$sleep_rem_isBAD, 
                           mean(msleep$sleep_rem, na.rm=TRUE), 
		           msleep$sleep_rem)


## ----novel, error=TRUE---------------------------------------------------
df <- data.frame(x=c('a', 'a', 'b', 'b', 'c', 'c'), 
                 y=1:6, 
                 stringsAsFactors=FALSE)
model <- lm(y~x, data=df)
newdata <- data.frame(x=c('a', 'b', 'c', 'd'), 
                      stringsAsFactors=FALSE)
tryCatch(
   predict(model, newdata=newdata), 
   error = function(e) print(strwrap(e)))


## ----vtreat, results='hide'----------------------------------------------
library("vtreat")
treatplan <- designTreatmentsN(df, 'x', 'y')
varnames <- treatplan$scoreFrame$varName[treatplan$scoreFrame$cod=="lev"]
newdata_treat <- prepare(treatplan, newdata, 
                         pruneSig=NULL, varRestriction=varnames)


## ----indicators1---------------------------------------------------------
print(newdata)


## ----indicators2---------------------------------------------------------
print(newdata_treat)


## ----zipcode-------------------------------------------------------------
set.seed(235)
Nz <- 25
zip <- paste0('z', format(1:Nz, justify="right"))
zip <- gsub(' ', '0', zip, fixed=TRUE)
zipval <- 1:Nz; names(zipval) <- zip
n <- 3; m <- Nz - n
p <- c(numeric(n) + (0.8/n), numeric(m) + 0.2/m)
N <- 1000
zipvar <- sample(zip, N, replace=TRUE, prob=p)
signal <-  zipval[zipvar] + rnorm(N)
d <- data.frame(zip=zipvar, 
                y=signal + rnorm(N))


## ----vzip, results="hide", warning=FALSE, message=FALSE------------------
library("vtreat")
treatplan <- designTreatmentsN(d, varlist="zip", outcome="y", verbose=FALSE)


## ----ziptreatplan--------------------------------------------------------
treatplan$meanY
scoreFrame <- treatplan$scoreFrame
scoreFrame[, c('varName', 'sig', 'extraModelDegrees', 'origName', 'code')]


## ----preparezip----------------------------------------------------------
vars <- scoreFrame$varName[!(scoreFrame$code %in% c("catP", "catD"))]
dtreated <- prepare(treatplan, d, pruneSig=NULL, 
                    varRestriction=vars)


## ----zipfig, echo=FALSE, message=FALSE, warning=FALSE, fig.width=6, fig.height=4----
# compare the impact coding (zip_catN) with the conditional outcome means
# to show that the impact model is the conditional outcome - global outcome
compf <- merge(data.frame(zip=zip), 
                data.frame(zip= d$zip, impact=dtreated$zip_catN, y=d$y), 
                by="zip", all.x=TRUE)
compf$meany <- treatplan$meanY
library("ggplot2")
ggplot(compf, aes(x=zip)) +
geom_point(aes(y=impact+meany), shape=23, fill="blue", size=3, alpha=0.7) +
geom_point(aes(y=y), alpha=0.3) +
coord_flip()


## ----zipmissing, warning=FALSE, message=FALSE----------------------------
N <- 100
zipvar <- sample(zip, N, replace=TRUE, prob=p)
signal <- zipval[zipvar] + rnorm(N)
d <- data.frame(zip=zipvar, 
                y=signal+rnorm(N))
length(unique(d$zip))
omitted <- setdiff(zip, unique(d$zip))
print(omitted)


## ----zipmissing2, warning=FALSE, message=FALSE---------------------------
treatplan <- designTreatmentsN(d, varlist="zip", outcome="y", verbose=FALSE)
dnew <- data.frame(zip = zip)
dtreated <- prepare(treatplan, dnew, pruneSig=NULL, 
                   varRestriction=vars)


## ----zipmissingprint-----------------------------------------------------
dtreated[dnew$zip %in% omitted, "zip_catN"]


## ----nested--------------------------------------------------------------
set.seed(2262)
nLev <- 500
n <- 3000
d <- data.frame(xBad1=sample(paste('level', 1:nLev, sep=''), n, replace=TRUE), 
                xBad2=sample(paste('level', 1:nLev, sep=''), n, replace=TRUE), 
                xGood1=sample(paste('level', 1:nLev, sep=''), n, replace=TRUE), 
                xGood2=sample(paste('level', 1:nLev, sep=''), n, replace=TRUE))
d$y <- (0.2*rnorm(nrow(d)) + 0.5*ifelse(as.numeric(d$xGood1)>nLev/2, 1, -1) +
        0.3*ifelse(as.numeric(d$xGood2)>nLev/2, 1, -1))>0
d$rgroup <- sample(c("cal", "train", "test"), nrow(d), replace=TRUE, 
                   prob=c(0.6, 0.2, 0.2))

plotRes <- function(d, predName, yName, title) {
  print(title)
  tab <- table(truth=d[[yName]], pred=d[[predName]]>0.5)
  print(tab)
  diag <- sum(vapply(seq_len(min(dim(tab))), 
                     function(i) tab[i, i], numeric(1)))
  acc <- diag/sum(tab)
  # depends on both truth and target being logicals
  # and FALSE ordered before TRUE
  sens <- tab[2, 2]/sum(tab[2, ])
  spec <- tab[1, 1]/sum(tab[1, ])
  print(paste('accuracy', format(acc, scientific=FALSE, digits=3)))
  print(paste('sensitivity', format(sens, scientific=FALSE, digits=3)))
  print(paste('specificity', format(spec, scientific=FALSE, digits=3)))
}


## ----naivesplit----------------------------------------------------------
dTrain <- d[d$rgroup!='test', , drop=FALSE]
dTest <- d[d$rgroup=='test', , drop=FALSE]
treatments <- vtreat::designTreatmentsC(dTrain, 
                     varlist = c('xBad1', 'xBad2', 'xGood1', 'xGood2'), 
                     outcomename='y', outcometarget=TRUE, 
                     verbose=FALSE)
dTrainTreated <- vtreat::prepare(treatments, dTrain, pruneSig=NULL)


## ----naivefit, warning=FALSE---------------------------------------------
m1 <- glm(y~xBad1_catB + xBad2_catB + xGood1_catB + xGood2_catB, 
          data=dTrainTreated, family=binomial(link='logit'))
print(summary(m1))


## ----nf2-----------------------------------------------------------------
dTrain$predM1 <- predict(m1, newdata=dTrainTreated, type='response')
plotRes(dTrain, 'predM1', 'y', 'model1 on train')


## ----nftest, message=FALSE, warning=FALSE--------------------------------
dTestTreated <- vtreat::prepare(treatments, dTest, pruneSig=NULL)
dTest$predM1 <- predict(m1, newdata=dTestTreated, type='response')
plotRes(dTest, 'predM1', 'y', 'model1 on test')


## ----calset1, message=FALSE, warning=FALSE-------------------------------
dCal <- d[d$rgroup=='cal', , drop=FALSE]
dTrain <- d[d$rgroup=='train', , drop=FALSE]
dTest <- d[d$rgroup=='test', , drop=FALSE]
treatments <- vtreat::designTreatmentsC(dCal, 
                varlist = c('xBad1', 'xBad2', 'xGood1', 'xGood2'), 
                outcomename='y', outcometarget=TRUE, 
                verbose=FALSE)
dTrainTreated <- vtreat::prepare(treatments, dTrain, 
                        pruneSig=NULL)
newvars <- setdiff(colnames(dTrainTreated), 'y')
m1 <- glm(y~xBad1_catB + xBad2_catB + xGood1_catB + xGood2_catB, 
          data=dTrainTreated, family=binomial(link='logit'))
dTrain$predM1 <- predict(m1, newdata=dTrainTreated, type='response')
print(summary(m1))


## ----calset2-------------------------------------------------------------
plotRes(dTrain, 'predM1', 'y', 'model1 on train')


## ----calset3-------------------------------------------------------------
dTestTreated <- vtreat::prepare(treatments, dTest, 
                                pruneSig=NULL)
dTest$predM1 <- predict(m1, newdata=dTestTreated, type='response')
plotRes(dTest, 'predM1', 'y', 'model1 on test')


## ----sigprune_data-------------------------------------------------------
set.seed(22451)
N <- 500
sigN <- rnorm(N)
noiseN <- rnorm(N)
Nlevels <- 100
zip <- paste0('z', format(1:Nlevels, justify="right"))
zip <- gsub(' ', '0', zip, fixed=TRUE)

zipval <- runif(Nlevels); names(zipval)=zip
sigC <- sample(zip, size=N, replace=TRUE)
noiseC <- sample(zip, size=N, replace=TRUE)

y <- sigN + zipval[sigC] + rnorm(N)
df <- data.frame(sN = sigN, nN=noiseN, 
                sC = sigC, nC=noiseC, y=y)


## ----sigprune_treat------------------------------------------------------
library("vtreat")
treatplan <- designTreatmentsN(df, 
                             varlist=setdiff(colnames(df), "y"), 
                             outcomename="y", 
                             verbose=FALSE)
sframe <- treatplan$scoreFrame
vars <- sframe$varName[!(sframe$code %in% c("catP", "catD"))]
sframe[sframe$varName %in% vars, 
       c("varName", "sig", "extraModelDegrees")]


## ----sigprune_plot, echo=FALSE, fig.width=6, fig.height=4----------------
sframe <- sframe[sframe$varName %in% vars, 
       c("varName", "sig", "extraModelDegrees", "origName")]
sframe$varType <- ifelse(sframe$origName %in% c("sN", "sC"), 
                        "signal", "noise")
library("ggplot2")
ggplot(sframe, aes(x=reorder(varName, sig), 
                   y=sig, color=varType, shape=varType)) +
  geom_pointrange(aes(ymin=0, ymax=sig)) +
  geom_hline(yintercept=1/length(vars), linetype=2, color="darkgray") +
  scale_color_manual(values=c("signal" = "#1b9e77", "noise" = "#d95f02")) +
  coord_flip()


## ----sigprune_prune------------------------------------------------------
pruneSig <- 1/length(vars)
dfTreat <- prepare(treatplan, df, pruneSig=pruneSig, 
                  varRestriction=vars)
head(dfTreat)

