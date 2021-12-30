library(tidyquant)
library(rugarch)
library(arrow)
options("getSymbols.warning4.0"=FALSE)
options("getSymbols.yahoo.warning"=FALSE)
# Downloading SP500 price using quantmod

getSymbols("^GSPC", from = '2007-06-11',
           to = "2012-04-10",warnings = FALSE,
           auto.assign = TRUE)

GSPC <- GSPC[,'GSPC.Close']


logr <- na.omit(diff(log(GSPC$GSPC.Close), lag=1))
#logr <- na.omit(exp(diff(log(GSPC$GSPC.Close), lag=1)))
length(logr)==1217
plot(logr*100)

spec.gjrGARCH = ugarchspec(variance.model=list(model="gjrGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0), include.mean=TRUE), distribution.model="std")
gjrGARCH <- ugarchfit(logr*100, spec=spec.gjrGARCH)
gjrGARCH

fixed.p <- list(mu = 0.056105, 
                omega = 0.023003, 
                alpha1 = 0.000000 , 
                beta1 = 0.895905, 
                gamma1 = 0.183166,
                shape = 6.930399)
spec.gjrGARCH = ugarchspec(variance.model=list(model="gjrGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0), include.mean=TRUE),fixed.pars = fixed.p, distribution.model="std")
X <- ugarchpath(spec.gjrGARCH,
                n.sim = 50, # simulated path length
                m.sim = 10**6 # number of paths to simulate
)
Y <- as.data.frame(X@path[["residSim"]])
Z <- as.data.frame(X@path[['sigmaSim']])

for (i in 1:4000) {
  X <- ugarchpath(spec.gjrGARCH,
                  n.sim = 50, # simulated path length
                  m.sim = 256 # number of paths to simulate
  )
  Y <- as.data.frame(X@path[["residSim"]])
  Z <- as.data.frame(X@path[['sigmaSim']])
  write_parquet(Y, paste0("v1/gjrpath", i, ".parquet"))
  write_parquet(Z, paste0("v1/gjrpath_sig", i, ".parquet"))
}



#https://stackoverflow.com/questions/30402253/how-do-i-read-a-parquet-in-r-and-convert-it-to-an-r-dataframe
#https://spark.apache.org/docs/1.6.1/api/R/write.parquet.html


