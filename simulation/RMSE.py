__author__ = 'bhash90'

import numpy as np;
def msd(predictions, targets):
   e = np.sqrt(np.mean((predictions-targets)**2))
   print(e)
   return e

target = np.array([0.9,3.9,7.9,15.79,25.1,36,50])
prediction = np.array([1,4,9,16,25,36,49])
msd(target,prediction)