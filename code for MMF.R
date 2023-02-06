rm(list=ls())
library(Hmisc);  #for data cleaning
library(grid);  #for plot
library(lattice); #for plot
library(ggplot2) #for plot
library(rms)  #for nomogram
set.seed(1234)    # seef for reproducibility
library(glmnet)  # for ridge regression
library(ResourceSelection)
library(rmda)

train <- read.csv('data.csv', header = T, row.names = 1, check.names = FALSE)
test <- read.csv('test.csv', header = T, row.names = 1)

dd=datadist(train)
options(datadist="dd") 
attach(train)

f1 <- lrm(Type~Borrmann_Type + Tumor_Location + cT_Stage, data = train, x=T, y=T) 

print(summary(f1)

coef_data <- as.data.frame(f1$coefficients,f1$coefficients)

colnames(coef_data) <- 'coefficients'

coef_data$term <- rownames(coef_data)


coef_data <- coef_data[-1,]


pre_data <-  as.data.frame(predict(f1,test))

colnames(pre_data) <- 'predict'


coef_data <-  coef_data[order(coef_data$term),]

nom <- nomogram(f1, fun= function(x) 1/(1+exp(-x)), # or fun=plogis
                lp=F,  fun.at=c(.05,seq(.1,.9,by=.1),.95), funlabel="PM Risk")
plot(nom,lwd.axis=1,xfrac=.2, cex.axis=1.5, cex.var=1.5, font.axis=1.5,col.axis='red')


library(riskRegression)
f1 <- lrm(formula=Type~Calibration, data = train, x=T, y=T)
cal2 <- calibrate(f1, cmethod='hare', method='boot', B=1000,data=train)
plot(1, type='n', xlim=c(0,1.0),ylim=c(0,1.0),
     main='Training Calibration Curve', xlab="Predicted Probability of PM", ylab="Actual Frequency of PM",
     legend=FALSE, subtitles=FALSE,font.lab=2,cex.axis=1.2,)
abline(0,1,col='skyblue3',lty=1,lwd=2)
lines(cal2[, c("predy", "calibrated.orig")], type='l', lty=6,cex.lab=1.2, lwd=3, col='red', pch=16)
legend(0.65,0.25,
       c('MMF model', 'Ideal'),
       lty = c(6,1),
       lwd = c(2,2),
       col = c('red', 'skyblue3'),
       bty = "n",
       text.font = c(2,2),
       seg.len = c(4,4)
       )
text(0.1, 0.9, "A", font=2, cex=2.5)

f1 <- lrm(formula=Type~Calibration, data = test, x=T, y=T)
cal2 <- calibrate(f1, cmethod='hare', method='boot', B=1000,data=test)
# plot(cal2, main='Test Calibration Curve',xlim=c(0,1.0),ylim=c(0,1.0))
plot(1, type='n', xlim=c(0,1.0),ylim=c(0,1.0),
     main='Testing Calibration Curve', xlab="Predicted Probability of PM", ylab="Actual Frequency of PM",
     legend=FALSE, subtitles=FALSE,font.lab=2,cex.axis=1.2)
abline(0,1,col='skyblue3',lty=1,lwd=2)
lines(cal2[, c("predy", "calibrated.orig")], type='l', lty=6, cex.lab=1.2, lwd=3, col='red', pch=16)
legend(0.65,0.25,
       c('MMF model', 'Ideal'),
       lty = c(6,1),
       lwd = c(2,2),
       col = c('red', 'skyblue3'),
       bty = "n",
       text.font = c(2,2),
       seg.len = c(4,4)
)
text(0.1, 0.9, "B", font=2, cex=2.5) 