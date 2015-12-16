library(fpp)
library(forecast)
options(error=NULL)

getMotionPridiction=function (tseries,horizon)
{
  currentIndex=length(tseries)
  avg=tseries[currentIndex];
  if(currentIndex>=3){
    u=(tseries[currentIndex]-tseries[currentIndex-2])/2  
    a=(tseries[currentIndex]-2*tseries[currentIndex-1]+tseries[currentIndex-2])
  }else{
     a=0;
     u=0;
  }
  return (avg+u*horizon+0.5*a*horizon*horizon)
}

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
      
      ArimaRE=c();
      ArimaAE=c()
      ArimaSE=c()
      ArimaSSE=c()
      ArimaSAE=c()
      Arimaresiduals=c();
      arimaSse=0;
      arimaSae=0;
      
      CurrentRE=c();
      CurrentAE=c()
      CurrentSE=c()
      CurrentSSE=c()
      CurrentSAE=c()
      Currentresiduals=c();
      currentSse=0;
      currenntSae=0;
      
      NNetRE=c();
      NNetAE=c()
      NNetSE=c()
      NNetSSE=c()
      NNetSAE=c()
      NNetresiduals=c();
      nnetSse=0;
      nnetSae=0;
     
      EnsambleRE=c();
      EnsambleAE=c()
      EnsambleSE=c()
      EnsambleSSE=c()
      EnsambleSAE=c()
      Ensambleresiduals=c();
      ensambletSse=0;
      ensambletSae=0;
     
     
      
      arimaPridicted=c();
      nnetPridicted=c();
      ensamblePridicted=c();
      currentPridicted=c();
      
      currentPridicted[1]=air[1];
      arimaPridicted[1]=air[1];
      nnetPridicted[1]=air[1]
      ensamblePridicted[1]=air[1]
      alpha=1;beta=1;
      for (i in 1:length(air)){
           dseries=c(dseries,air[i])
           tseries=ts(dseries)
           
           arimaPridicted[i+1]=getArimaPridiction(tseries,1)
           nnetPridicted[i+1]=getNNetPridiction(tseries,1)
           
           
           Arimaresiduals[i]=(arimaPridicted[i]-tseries[i])
           ArimaAE[i]=abs(Arimaresiduals[i])
           ArimaSE[i]= ArimaAE[i]*ArimaAE[i];
           arimaSse=arimaSse+ ArimaSE[i]
           arimaSae=arimaSae+ ArimaAE[i] 
           ArimaRE[i]=ArimaAE[i]/tseries[i]
           ArimaSSE[i]=arimaSse;
           ArimaSAE[i]=arimaSae;
           
           NNetresiduals[i]=(nnetPridicted[i]-tseries[i])
           NNetAE[i]=abs( NNetresiduals[i])
           NNetSE[i]=  NNetAE[i]* NNetAE[i];
           nnetSse=nnetSse+  NNetSE[i]
           nnetSae= nnetSae+  NNetAE[i] 
           NNetRE[i]= NNetAE[i]/tseries[i]
           NNetSSE[i]=nnetSse;
           NNetSAE[i]=nnetSae;
           
           
           
           
           if(i==1 || ArimaAE[i-1]==0 || NNetAE[i-1]==0)
           {  alpha=1
              beta=1
           }else
           {       if(NNetresiduals[i]*Arimaresiduals[i]<0){
                    alpha=NNetAE[i];
                    beta=ArimaAE[i]             
                   }
                   else
                   {
                     if(ArimaAE[i]<NNetAE[i]){
                       alpha=1;
                       beta=0 
                     } 
                     else
                     {
                       alpha=0;
                       beta=1; 
                     }
                   }
           }
           ensamblePridicted[i+1]=((alpha*arimaPridicted[i+1]+beta*nnetPridicted[i+1])/(alpha+beta))
           currentPridicted[i+1]= getMotionPridiction(tseries,1)
                           
           
             
         
           
           Ensambleresiduals[i]=(ensamblePridicted[i]-tseries[i])
           EnsambleAE[i]=abs( Ensambleresiduals[i])
           EnsambleSE[i]=  EnsambleAE[i]* EnsambleAE[i];
           ensambletSse=ensambletSse+  EnsambleSE[i]
           ensambletSae= ensambletSae+  EnsambleAE[i] 
           EnsambleRE[i]= EnsambleAE[i]/tseries[i]
           EnsambleSSE[i]=ensambletSse;
           EnsambleSAE[i]=ensambletSae;
           
          
           Currentresiduals[i]=(currentPridicted[i]-tseries[i])
           CurrentAE[i]=abs(Currentresiduals[i])
           CurrentSE[i]= CurrentAE[i]*CurrentAE[i];
           currentSse=currentSse+ CurrentSE[i]
           currenntSae=currenntSae+ CurrentAE[i] 
           CurrentRE[i]=CurrentAE[i]/tseries[i]
           CurrentSSE[i]=currentSse;
           CurrentSAE[i]=currenntSae;
           
      }
      ds=cbind(dseries,currentPridicted,arimaPridicted,nnetPridicted,ensamblePridicted,CurrentAE,ArimaAE,NNetAE,EnsambleAE,CurrentRE,ArimaRE,NNetRE,EnsambleRE)
      View(ds)
      old.par <- par(mfrow=c(2,2 ))
      plot(ts(ds[,2],start=start(air)),col=2,plot.type = "s",ylab="", main="Real data vs Motion Equation") 
      lines(ts(ds[,1],start=start(air)),col=1)
      
      plot(ts(ds[,3],start=start(air)),col=3,plot.type = "s",ylab="A", main="Real data vs ARIMA prediction")
      lines(ts(ds[,1],start=start(air)),col=1)
      
      plot(ts(ds[,4],start=start(air)),col=4,plot.type = "s",ylab="", main="Real data vs Neural Network Prediction")
      lines(ts(ds[,1],start=start(air)),col=1)
      
      plot(ts(ds[,5],start=start(air)),col=5,plot.type = "s",ylab="", main="Real data vs Ensamble Prediction")
      lines(ts(ds[,1],start=start(air)),col=1)
      
    
}


pridiction(oil)
#air, euretail,sunspotarea,oil,ausair,austourists(nnet)
