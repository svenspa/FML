library(tidyquant)
library(rugarch)
options("getSymbols.warning4.0"=FALSE)
options("getSymbols.yahoo.warning"=FALSE)
# Downloading SP500 price using quantmod

getSymbols("^GSPC", from = '2013-03-04',
           to = "2018-01-02",warnings = FALSE,
           auto.assign = TRUE)

GSPC <- GSPC[,'GSPC.Close']


logr <- na.omit(diff(log(GSPC$GSPC.Close), lag=1))
length(logr)==1217
plot(logr)

spec.gjrGARCH = ugarchspec(variance.model=list(model="gjrGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0), include.mean=TRUE), distribution.model="std")
gjrGARCH <- ugarchfit(logr, spec=spec.gjrGARCH)
gjrGARCH
