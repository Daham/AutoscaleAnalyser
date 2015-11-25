library(fpp)
library(forecast)
options(error=NULL)
getArimaPridiction=function(tseries,horizon){
    arimaFit=auto.arima(tseries)
    fmodel=forecast(arimaFit,h=horizon)
    return (fmodel$mean[1])
  }
  
getExpPridiction=function(tseries,horizon){
  #if(length(tseries)<7)
  #tseries=ts(c(tseries[1],tseries[1]-1,tseries[1]-2,tseries[1]-3,tseries[1],tseries[1]))
    
  #expFit=ets(tseries, model="ZZZ", damped=NULL, alpha=NULL, beta=NULL,
       #        gamma=NULL, phi=NULL, additive.only=FALSE, lambda=NULL,
        #       lower=c(rep(0.0001,3), 0.8), upper=c(rep(0.9999,3),0.98),
         #      opt.crit=c("lik","amse","mse","sigma","mae"), nmse=3,
           #    bounds=c("both","usual","admissible"),
             #  ic=c("aicc","aic","bic"), restrict=TRUE)
  #expFit=holt(tseries,exponential=TRUE,damped=TRUE)
  expFit=ses(tseries,alpha = 0.96,initial="optimal", h=1)
  fmodel=forecast(expFit,h=horizon)
  return (expFit$mean)
}


getNNetPridiction=function(tseries,horizon){
  
  if(length(tseries)==1)
    tseries=ts(c(tseries[1],tseries[1]-1))
  
  nnetFit=nnetar(tseries,lambda=0)
  fmodel=forecast(nnetFit,h=horizon)
  return (fmodel$mean[1])
 
}

pridiction=function (air){
      dseries=c();
      tseries=ts();     
      
      RE=c();
      AE=c()
      SE=c()
      SSE=c()
      SAE=c()
      residuals=c();
      
      sse=0;
      sae=0;
      pridicted=c();
      pridicted[1]=air[1];
      
      for (i in 1:length(air)){
           dseries=c(dseries,air[i])
           tseries=ts(dseries)
            pridicted[i+1]=getExpPridiction(tseries,1)
            residuals[i]=(pridicted[i]-tseries[i])
           AE[i]=abs(residuals[i])
           SE[i]= AE[i]*AE[i];
           sse=sse+SE[i]
           sae=sae+AE[i] 
           RE[i]=AE[i]/tseries[i]
           SSE[i]=sse;
           SAE[i]=sae;
      }
      ds=cbind(dseries,pridicted,AE,RE,SE,SAE,SSE)
      #View(ds)
     # fmodel=forecast(auto.arima(air),h=1)
      #a=ts(pridicted,start=1875)
     #plot(fmodel,col=2,plot.type = "s")
     #lines(a,col = 3)
      #lines(air,col=4)
     return (ds)
}


ds=pridiction(air)
View(ds)
