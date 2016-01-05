
library(forecast);
options(error=NULL);
getArimaPrediction=function(tseries,horizon) {
  arimaFit=auto.arima(tseries);
  fmodel=forecast(arimaFit,h=horizon);
  return (fmodel$mean[1:horizon]);
}
getNNetPrediction=function(tseries,horizon) {
  if(length(tseries)==1)
    tseries=ts(c(tseries[1],tseries[1]-1));

  nnetFit=nnetar(tseries,lambda=0);
  fmodel=forecast(nnetFit,h=horizon);
  return (fmodel$mean[1:horizon]);
}


prediction=function (tsdata, horizon=1) {
    arimaWindow = getArimaPrediction(tsdata, horizon);
    nnetWindow = getNNetPrediction(tsdata, horizon);
    alpha=1;
    beta=1;
    ensembleWindow=c();

    for (j in 1:horizon) {
      ensembleWindow[j] = ((alpha*arimaWindow[j]+beta*nnetWindow[j])/(alpha+beta));
    }
    return(ensembleWindow);
}