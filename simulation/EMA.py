__author__ = 'bhash90'
import numpy as np;
def ema(values, window):
    weights = np.exp(np.linspace(-1.,0.,window))
    weights /= weights.sum() #weights is a window sized array
    print(weights)
    a = np.convolve(values,weights)[0:len(values)]
    print(a)
    a[:window] = a[window]
    print(a)
    return a;

a = ema ([1,5,7,21,19,15], 3)
