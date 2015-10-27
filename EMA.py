__author__ = 'bhash90'
import numpy as np;
import matplotlib.pyplot as plt
def ema(values, window):
    weights = np.exp(np.linspace(-1.,0.,window))
    weights /= weights.sum()

    a = np.convolve(values,weights)[:len(values)]
    a[:window] = a[window]
    return a;
